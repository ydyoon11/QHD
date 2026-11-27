#include "Simulation.hpp"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_FFT.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace {

using namespace amrex;
namespace py = pybind11;

Geometry make_geometry(int nx, int ny, double lx, double ly) {
    IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    IntVect dom_hi(AMREX_D_DECL(nx - 1, ny - 1, 0));
    Box domain(dom_lo, dom_hi);
    RealBox real_box({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(lx, ly, 1.0)});
    Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 0)};
    return Geometry(domain, &real_box, CoordSys::cartesian, is_periodic.data());
}

BoxArray make_box_array(const Box& box) {
    BoxArray ba(box);
    ba.maxSize(std::clamp(std::min(box.length(0), box.length(1)), 32, 128));
    return ba;
}

void reset_output_dir(const std::string& output_dir) {
    if (ParallelDescriptor::IOProcessor()) {
        std::error_code ec;
        std::filesystem::remove_all(output_dir, ec);
        ec.clear();
        std::filesystem::create_directories(output_dir, ec);
        if (ec) {
            throw std::runtime_error("Failed to create output directory '" + output_dir + "': " + ec.message());
        }
    }
    ParallelDescriptor::Barrier();
}

std::string quote_shell_arg(const std::string& arg) {
    std::string quoted;
    quoted.reserve(arg.size() + 2);
    quoted.push_back('"');
    for (char c : arg) {
        if (c == '"' || c == '\\' || c == '$' || c == '`') {
            quoted.push_back('\\');
        }
        quoted.push_back(c);
    }
    quoted.push_back('"');
    return quoted;
}

int decode_process_exit_code(int rc) {
#ifdef _WIN32
    return rc;
#else
    if (rc == -1) {
        return rc;
    }
    if (WIFEXITED(rc)) {
        return WEXITSTATUS(rc);
    }
    return rc;
#endif
}

std::filesystem::path make_unique_temp_path(const std::string& prefix, int step = -1) {
    const auto ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::ostringstream name;
    name << prefix << "_" << ParallelDescriptor::NProcs() << "_" << ticks;
    if (step >= 0) {
        name << "_step" << step;
    }
    return std::filesystem::temp_directory_path() / name.str();
}

std::filesystem::path make_shared_temp_path(const std::string& prefix, int step = -1) {
    std::string path_string;
    if (ParallelDescriptor::IOProcessor()) {
        path_string = make_unique_temp_path(prefix, step).string();
    }

    int size = static_cast<int>(path_string.size());
    ParallelDescriptor::Bcast(&size, 1, ParallelDescriptor::IOProcessorNumber());

    Vector<char> buffer(static_cast<std::size_t>(size) + 1, '\0');
    if (ParallelDescriptor::IOProcessor()) {
        std::copy(path_string.begin(), path_string.end(), buffer.begin());
    }
    ParallelDescriptor::Bcast(buffer.data(), buffer.size(), ParallelDescriptor::IOProcessorNumber());

    if (!ParallelDescriptor::IOProcessor()) {
        path_string.assign(buffer.data(), static_cast<std::size_t>(size));
    }
    return std::filesystem::path(path_string);
}

std::vector<double> read_npy_2d(const std::string& path, int expected_ny, int expected_nx) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open numpy array file: " + path);
    }

    char magic[6];
    in.read(magic, 6);
    if (in.gcount() != 6 || std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Unsupported .npy file header in " + path);
    }

    unsigned char major = 0;
    unsigned char minor = 0;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);
    if (!in) {
        throw std::runtime_error("Failed to read .npy version from " + path);
    }

    std::uint32_t header_len = 0;
    if (major == 1) {
        std::uint16_t len16 = 0;
        in.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else if (major == 2 || major == 3) {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("Unsupported .npy version in " + path);
    }

    std::string header(header_len, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!in) {
        throw std::runtime_error("Failed to read .npy header payload from " + path);
    }

    if (header.find("'fortran_order': True") != std::string::npos) {
        throw std::runtime_error("Fortran-order .npy arrays are not supported: " + path);
    }

    std::smatch shape_match;
    const std::regex shape_re(R"('shape':\s*\((\d+)\s*,\s*(\d+)\s*,?\))");
    if (!std::regex_search(header, shape_match, shape_re)) {
        throw std::runtime_error("Only 2D .npy arrays are supported: " + path);
    }

    const int ny = std::stoi(shape_match[1].str());
    const int nx = std::stoi(shape_match[2].str());
    if (ny != expected_ny || nx != expected_nx) {
        throw std::runtime_error("Array shape mismatch in " + path + ": expected (" +
                                 std::to_string(expected_ny) + ", " + std::to_string(expected_nx) +
                                 "), got (" + std::to_string(ny) + ", " + std::to_string(nx) + ")");
    }

    const bool is_f8 = header.find("'descr': '<f8'") != std::string::npos ||
                       header.find("\"descr\": \"<f8\"") != std::string::npos;
    const bool is_f4 = header.find("'descr': '<f4'") != std::string::npos ||
                       header.find("\"descr\": \"<f4\"") != std::string::npos;
    if (!is_f8 && !is_f4) {
        throw std::runtime_error("Only little-endian float32/float64 .npy arrays are supported: " + path);
    }

    std::vector<double> out(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
    if (is_f8) {
        in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(double)));
    } else {
        std::vector<float> tmp(out.size());
        in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(tmp.size() * sizeof(float)));
        for (std::size_t i = 0; i < tmp.size(); ++i) {
            out[i] = tmp[i];
        }
    }

    if (!in) {
        throw std::runtime_error("Failed to read array payload from " + path);
    }

    return out;
}

void ensure_finite(const char* label, const MultiFab& mf) {
    if (mf.contains_nan()) {
        throw std::runtime_error(std::string(label) + " contains NaN");
    }
}

void ensure_finite_field(const char* label, const vecops::VectorField2D& field) {
    ensure_finite((std::string(label) + ".x").c_str(), field.comp[vecops::X]);
    ensure_finite((std::string(label) + ".y").c_str(), field.comp[vecops::Y]);
    ensure_finite((std::string(label) + ".z").c_str(), field.comp[vecops::Z]);
}

void ensure_finite_tensor(const char* label, const vecops::SymmetricTensorField2D& field) {
    static constexpr std::array<const char*, 6> names = {"xx", "xy", "xz", "yy", "yz", "zz"};
    for (int n = 0; n < 6; ++n) {
        ensure_finite((std::string(label) + "." + names[n]).c_str(), field.comp[n]);
    }
}

} // namespace

// Runs the config's Python pressure closures, one per species:
//   electrons: `pressure_closure`      (+ optional `pressure_closure_dependencies`)
//   ions:      `ion_pressure_closure`  (+ optional `ion_pressure_closure_dependencies`), ions = 1 only
// Both closures get a ctx with the SAME keys, so one closure body works for either species:
//   ctx key   electron closure   ion closure
//   "b"       B                  B (shared)
//   "q"       Q_e                Q_i
//   "u"       u_e                u_i
//   "p"       P_e (last value)   P_i (last value)
//   "p_out"   P_e, written       P_i, written
//   "species" "electron"         "ion"
// plus the grid keys (nx ny lx ly dx dy x1d y1d coords) and t, dt, step, identical for both.
class PressureClosureEngine {
public:
    PressureClosureEngine(const std::string& namelist_path, const SimulationConfig& cfg)
        : guard_(std::make_unique<py::scoped_interpreter>()),
          nx_(cfg.nx),
          ny_(cfg.ny),
          prob_hi_x_(cfg.lx),
          prob_hi_y_(cfg.ly) {
        py::gil_scoped_acquire gil;
        // Turn Python errors into plain C++ errors here, while the interpreter is alive, so the
        // caller can print the message after this object (and the interpreter) is gone.
        try {
            numpy_ = py::module_::import("numpy");
            x1d_ = make_center_line(nx_, cfg.lx / cfg.nx);
            y1d_ = make_center_line(ny_, cfg.ly / cfg.ny);
            coords_ = build_coordinate_cache(cfg.lx, cfg.ly);

            py::module_ runpy = py::module_::import("runpy");
            py::dict init_globals;
            init_globals["__file__"] = namelist_path;
            init_globals["np"] = numpy_;
            init_globals["nx"] = cfg.nx;
            init_globals["ny"] = cfg.ny;
            init_globals["lx"] = cfg.lx;
            init_globals["ly"] = cfg.ly;
            init_globals["dx"] = cfg.lx / cfg.nx;
            init_globals["dy"] = cfg.ly / cfg.ny;
            init_globals["x1d"] = x1d_;
            init_globals["y1d"] = y1d_;
            init_globals["t"] = 0.0;
            init_globals["dt"] = cfg.dt;
            init_globals["step"] = 0;
            if (cfg.ions) {
                // M and nu_i are separate config globals, read by the ion closure the way the
                // electron closure reads nu. Seeded with the parsed values so they exist even if the
                // config leaves nu_i to its default; a config that assigns them overrides the seed.
                init_globals["mass_ratio"] = cfg.mass_ratio;
                init_globals["nu_i"] = cfg.nu_i;
            }

            py::dict globals = runpy.attr("run_path")(namelist_path, py::arg("init_globals") = init_globals).cast<py::dict>();
            load_closure(globals, "pressure_closure", "pressure_closure_dependencies", "electron",
                         slot(Species::electron));
            if (cfg.ions) {
                load_closure(globals, "ion_pressure_closure", "ion_pressure_closure_dependencies", "ion",
                             slot(Species::ion));
            }
        } catch (const py::error_already_set& e) {
            throw std::runtime_error(e.what());
        }
    }

    bool has_closure(Species species) const noexcept {
        return static_cast<bool>(slot(species).fn);
    }

    bool needs_b(Species species) const noexcept { return slot(species).need_b; }
    bool needs_q(Species species) const noexcept { return slot(species).need_q; }
    bool needs_u(Species species) const noexcept { return slot(species).need_u; }
    bool needs_p(Species species) const noexcept { return slot(species).need_p; }

    void evaluate(Species species,
                  Real time,
                  Real dt,
                  int step,
                  const std::array<MultiFab, 3>& b_global,
                  const std::array<MultiFab, 3>& q_global,
                  const std::array<MultiFab, 3>& u_global,
                  const std::array<MultiFab, 6>& p_in_global,
                  std::array<MultiFab, 6>& p_out_global) {
        ClosureSlot& s = slot(species);
        if (!has_closure(species)) {
            throw std::runtime_error("PressureClosureEngine::evaluate called without a closure for this species");
        }

        py::gil_scoped_acquire gil;

        try {
            ensure_ctx_bound(s, b_global, q_global, u_global, p_in_global, p_out_global);
            s.ctx["t"] = time;
            s.ctx["dt"] = dt;
            s.ctx["step"] = step;

            py::object result_obj = s.fn(s.ctx);
            if (result_obj.is_none()) {
                return;
            }
            if (!py::isinstance<py::dict>(result_obj)) {
                throw std::runtime_error(s.name + "(ctx) must return a dict or None for in-place output");
            }

            py::dict result = result_obj.cast<py::dict>();
            for (int n = 0; n < 6; ++n) {
                load_component_into_multifab(s, result, component_name(n), p_out_global[n]);
            }
        } catch (const py::error_already_set& e) {
            throw std::runtime_error(s.name + ": " + e.what());
        }
    }

private:
    // One species' closure and its ctx. The ctx holds NumPy views of that species' gather buffers,
    // built on the first call and reused; only t, dt and step change between calls.
    struct ClosureSlot {
        std::string name;               // the config's name for it, e.g. "ion_pressure_closure"
        const char* species = "";       // ctx["species"]
        py::object fn;                  // empty unless the config defines the closure
        bool need_b = true;
        bool need_q = true;
        bool need_u = true;
        bool need_p = true;
        py::dict ctx;
        py::dict b_dict;
        py::dict q_dict;
        py::dict u_dict;
        py::dict p_dict;
        py::dict p_out_dict;
        bool ctx_bound = false;
    };

    ClosureSlot& slot(Species species) { return slots_[static_cast<std::size_t>(species)]; }
    const ClosureSlot& slot(Species species) const { return slots_[static_cast<std::size_t>(species)]; }

    void load_closure(const py::dict& globals,
                      const char* fn_name,
                      const char* deps_name,
                      const char* species,
                      ClosureSlot& s) {
        s.name = fn_name;
        s.species = species;
        parse_dependencies(globals, deps_name, s);
        if (globals.contains(fn_name)) {
            py::object candidate = globals[fn_name];
            if (!PyCallable_Check(candidate.ptr())) {
                throw std::runtime_error(std::string(fn_name) + " exists but is not callable");
            }
            s.fn = candidate;
        }
    }

    static const char* component_name(int n) {
        static constexpr std::array<const char*, 6> names = {"xx", "xy", "xz", "yy", "yz", "zz"};
        return names[static_cast<std::size_t>(n)];
    }

    py::array_t<Real> make_center_line(int n, Real h) const {
        py::array_t<Real> out(static_cast<py::ssize_t>(n));
        auto arr = out.mutable_unchecked<1>();
        for (int i = 0; i < n; ++i) {
            arr(i) = (static_cast<Real>(i) + Real(0.5)) * h;
        }
        return out;
    }

    std::pair<py::array_t<Real>, py::array_t<Real>> make_mesh(const std::vector<Real>& x,
                                                               const std::vector<Real>& y) const {
        py::array_t<Real> xgrid({static_cast<py::ssize_t>(y.size()), static_cast<py::ssize_t>(x.size())});
        py::array_t<Real> ygrid({static_cast<py::ssize_t>(y.size()), static_cast<py::ssize_t>(x.size())});
        auto xarr = xgrid.mutable_unchecked<2>();
        auto yarr = ygrid.mutable_unchecked<2>();
        for (py::ssize_t j = 0; j < static_cast<py::ssize_t>(y.size()); ++j) {
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(x.size()); ++i) {
                xarr(j, i) = x[static_cast<std::size_t>(i)];
                yarr(j, i) = y[static_cast<std::size_t>(j)];
            }
        }
        return {std::move(xgrid), std::move(ygrid)};
    }

    py::dict make_named_mesh(const std::vector<Real>& x, const std::vector<Real>& y) const {
        auto [xgrid, ygrid] = make_mesh(x, y);
        py::dict out;
        out["x"] = std::move(xgrid);
        out["y"] = std::move(ygrid);
        return out;
    }

    py::dict build_coordinate_cache(Real lx, Real ly) const {
        const Real dx = lx / nx_;
        const Real dy = ly / ny_;

        std::vector<Real> x_cc(nx_);
        std::vector<Real> y_cc(ny_);
        std::vector<Real> x_edge_x(nx_);
        std::vector<Real> y_edge_x(ny_);
        std::vector<Real> x_edge_y(nx_);
        std::vector<Real> y_edge_y(ny_);
        std::vector<Real> x_node(nx_);
        std::vector<Real> y_node(ny_);
        for (int i = 0; i < nx_; ++i) {
            x_cc[i] = (static_cast<Real>(i) + Real(0.5)) * dx;
            x_edge_x[i] = (static_cast<Real>(i) + Real(0.5)) * dx;
            x_edge_y[i] = static_cast<Real>(i) * dx;
            x_node[i] = static_cast<Real>(i) * dx;
        }
        for (int j = 0; j < ny_; ++j) {
            y_cc[j] = (static_cast<Real>(j) + Real(0.5)) * dy;
            y_edge_x[j] = static_cast<Real>(j) * dy;
            y_edge_y[j] = (static_cast<Real>(j) + Real(0.5)) * dy;
            y_node[j] = static_cast<Real>(j) * dy;
        }

        py::dict coords;
        coords["Bx"] = make_named_mesh(x_edge_x, y_edge_x);
        coords["By"] = make_named_mesh(x_edge_y, y_edge_y);
        coords["Bz"] = make_named_mesh(x_cc, y_cc);
        coords["Qx"] = make_named_mesh(x_edge_x, y_edge_x);
        coords["Qy"] = make_named_mesh(x_edge_y, y_edge_y);
        coords["Qz"] = make_named_mesh(x_cc, y_cc);
        coords["Ux"] = make_named_mesh(x_edge_x, y_edge_x);
        coords["Uy"] = make_named_mesh(x_edge_y, y_edge_y);
        coords["Uz"] = make_named_mesh(x_cc, y_cc);
        coords["Pxx"] = make_named_mesh(x_node, y_node);
        coords["Pxy"] = make_named_mesh(x_cc, y_cc);
        coords["Pxz"] = make_named_mesh(x_edge_y, y_edge_y);
        coords["Pyy"] = make_named_mesh(x_node, y_node);
        coords["Pyz"] = make_named_mesh(x_edge_x, y_edge_x);
        coords["Pzz"] = make_named_mesh(x_cc, y_cc);
        return coords;
    }

    void parse_dependencies(const py::dict& globals, const char* deps_name, ClosureSlot& s) {
        s.need_b = true;
        s.need_q = true;
        s.need_u = true;
        s.need_p = true;

        if (!globals.contains(deps_name)) {
            return;
        }

        s.need_b = false;
        s.need_q = false;
        s.need_u = false;
        s.need_p = false;

        auto add_dependency = [&](const std::string& dep) {
            std::string lower = dep;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (lower == "b") {
                s.need_b = true;
            } else if (lower == "q") {
                s.need_q = true;
            } else if (lower == "u") {
                s.need_u = true;
            } else if (lower == "p") {
                s.need_p = true;
            } else {
                throw std::runtime_error("Unknown entry in " + std::string(deps_name) + ": " + dep);
            }
        };

        py::object deps = globals[deps_name];
        if (py::isinstance<py::str>(deps)) {
            add_dependency(deps.cast<std::string>());
            return;
        }

        for (py::handle item : deps) {
            add_dependency(py::cast<std::string>(item));
        }
    }

    const FArrayBox& single_local_fab(const MultiFab& mf) const {
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            return mf[mfi];
        }
        throw std::runtime_error("Expected gathered MultiFab with one local FAB on the IO rank");
    }

    FArrayBox& single_local_fab(MultiFab& mf) const {
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            return mf[mfi];
        }
        throw std::runtime_error("Expected gathered MultiFab with one local FAB on the IO rank");
    }

    py::array array_view(const MultiFab& mf) const {
        const FArrayBox& fab = single_local_fab(mf);
        return py::array(
            py::dtype::of<Real>(),
            {static_cast<py::ssize_t>(ny_), static_cast<py::ssize_t>(nx_)},
            {static_cast<py::ssize_t>(sizeof(Real) * nx_), static_cast<py::ssize_t>(sizeof(Real))},
            const_cast<Real*>(fab.dataPtr()),
            py::none());
    }

    py::dict make_vector_dict(const std::array<MultiFab, 3>& field) const {
        py::dict out;
        out["x"] = array_view(field[0]);
        out["y"] = array_view(field[1]);
        out["z"] = array_view(field[2]);
        return out;
    }

    py::dict make_pressure_dict(const std::array<MultiFab, 6>& field) const {
        py::dict out;
        out["xx"] = array_view(field[0]);
        out["xy"] = array_view(field[1]);
        out["xz"] = array_view(field[2]);
        out["yy"] = array_view(field[3]);
        out["yz"] = array_view(field[4]);
        out["zz"] = array_view(field[5]);
        return out;
    }

    void ensure_ctx_bound(ClosureSlot& s,
                          const std::array<MultiFab, 3>& b_global,
                          const std::array<MultiFab, 3>& q_global,
                          const std::array<MultiFab, 3>& u_global,
                          const std::array<MultiFab, 6>& p_in_global,
                          std::array<MultiFab, 6>& p_out_global) {
        if (s.ctx_bound) {
            return;
        }

        s.ctx = py::dict();
        s.ctx["species"] = s.species;
        s.ctx["nx"] = nx_;
        s.ctx["ny"] = ny_;
        s.ctx["lx"] = prob_hi_x_;
        s.ctx["ly"] = prob_hi_y_;
        s.ctx["dx"] = prob_hi_x_ / nx_;
        s.ctx["dy"] = prob_hi_y_ / ny_;
        s.ctx["x1d"] = x1d_;
        s.ctx["y1d"] = y1d_;
        s.ctx["coords"] = coords_;

        if (s.need_b) {
            s.b_dict = make_vector_dict(b_global);
            s.ctx["b"] = s.b_dict;
        }
        if (s.need_q) {
            s.q_dict = make_vector_dict(q_global);
            s.ctx["q"] = s.q_dict;
        }
        if (s.need_u) {
            s.u_dict = make_vector_dict(u_global);
            s.ctx["u"] = s.u_dict;
        }
        if (s.need_p) {
            s.p_dict = make_pressure_dict(p_in_global);
            s.ctx["p"] = s.p_dict;
        }

        s.p_out_dict = make_pressure_dict(p_out_global);
        s.ctx["p_out"] = s.p_out_dict;
        s.ctx_bound = true;
    }

    py::object lookup_result_component(const ClosureSlot& s, const py::dict& result, const char* name) const {
        std::string lower(name);
        std::string upper = lower;
        std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) {
            return static_cast<char>(std::toupper(c));
        });

        const std::array<std::string, 6> keys = {
            lower,
            upper,
            std::string("p") + lower,
            std::string("P") + lower,
            std::string("p") + upper,
            std::string("P") + upper,
        };
        for (const auto& key : keys) {
            if (result.contains(py::str(key))) {
                return result[py::str(key)];
            }
        }

        throw std::runtime_error(s.name + " result is missing component `" + std::string(name) + "`");
    }

    void load_component_into_multifab(const ClosureSlot& s, const py::dict& result, const char* name, MultiFab& mf) const {
        py::object value = lookup_result_component(s, result, name);
        FArrayBox& fab = single_local_fab(mf);
        Real* data = fab.dataPtr();
        const std::size_t count = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_);

        if (py::isinstance<py::float_>(value) || py::isinstance<py::int_>(value)) {
            std::fill(data, data + count, value.cast<Real>());
            return;
        }

        py::object broadcast = numpy_.attr("broadcast_to")(value, py::make_tuple(ny_, nx_));
        py::array_t<Real, py::array::c_style | py::array::forcecast> arr(broadcast);
        if (arr.ndim() != 2 || arr.shape(0) != ny_ || arr.shape(1) != nx_) {
            throw std::runtime_error(s.name + " component `" + std::string(name) +
                                     "` returned an unexpected shape");
        }
        std::copy(arr.data(), arr.data() + count, data);
    }

    std::unique_ptr<py::scoped_interpreter> guard_;
    py::object numpy_;
    int nx_ = 0;
    int ny_ = 0;
    Real prob_hi_x_ = 0.0;
    Real prob_hi_y_ = 0.0;
    py::array_t<Real> x1d_;
    py::array_t<Real> y1d_;
    py::dict coords_;
    std::array<ClosureSlot, 2> slots_;  // indexed by Species: electron, ion
};

Simulation::Simulation(const SimulationConfig& cfg)
    : cfg_(cfg) {
    const Geometry geom = make_geometry(cfg_.nx, cfg_.ny, cfg_.lx, cfg_.ly);
    define_level(geom, make_box_array(geom.Domain()));
    helmholtz_fft_ = std::make_unique<FFT::R2C<Real>>(level_.geom.Domain());
    initialize_state();

    if (cfg_.ions && init_has_q_) {
        throw std::runtime_error("ions = 1: initializing from Q is not supported; give Bx, By, Bz");
    }

    if (!init_has_b_ && init_has_q_) {
        refresh_derived_state(level_.state, level_.stage, 0.0_rt, cfg_.dt, 0);
    } else if (init_has_b_ && !init_has_q_) {
        compute_q_from_b(level_.state.b, level_.state.u, level_.state.q, level_.stage);
    } else if (init_has_b_ && init_has_q_) {
        compute_u_from_b(level_.state.b, level_.state.u, level_.stage);
        fill_level_ghosts(level_.state.u);
    } else {
        throw std::runtime_error("Initialization did not provide either B or Q");
    }

    if (cfg_.ions) {
        // Ions start at rest: u_i = 0, so omega_i = 0 and Q_i = M*omega_i + B = B.
        // Then u_e = u_i - curl B = -curl B, so the q and u computed above are already Q_e and u_e.
        vecops::copy(level_.state.b, level_.state.q_i);
        fill_level_ghosts(level_.state.q_i);
        vecops::set_val(level_.state.u_i, 0.0_rt);
        if (ParallelDescriptor::IOProcessor()) {
            Print() << std::setprecision(6)
                    << "[ions] two-species QHD on: M = " << cfg_.mass_ratio
                    << ", nu_e = " << cfg_.nu << ", nu_i = " << cfg_.nu_i << "\n";
        }
    }

    std::string pressure_namelist;
    if (std::filesystem::path(cfg_.config_path).extension() == ".py") {
        pressure_namelist = cfg_.config_path;
    } else if (!cfg_.init_python_namelist.empty()) {
        pressure_namelist = cfg_.init_python_namelist;
    }

    if (!pressure_namelist.empty()) {
        // closure_status[0]: 0 = engine kept, 1 = setup failed, 2 = the file defines no closure.
        // closure_status[1], [2]: 1 if the file defines the electron / ion closure.
        int closure_status[3] = {2, 0, 0};
        if (ParallelDescriptor::IOProcessor()) {
            try {
                auto engine = std::make_unique<PressureClosureEngine>(pressure_namelist, cfg_);
                closure_status[1] = engine->has_closure(Species::electron) ? 1 : 0;
                closure_status[2] = engine->has_closure(Species::ion) ? 1 : 0;
                if (closure_status[1] != 0 || closure_status[2] != 0) {
                    pressure_engine_ = std::move(engine);
                    closure_status[0] = 0;
                }
            } catch (const std::exception& ex) {
                Print() << "[closure] Python setup error: " << ex.what() << "\n";
                closure_status[0] = 1;
            } catch (...) {
                closure_status[0] = 1;
            }
        }
        ParallelDescriptor::Bcast(closure_status, 3, ParallelDescriptor::IOProcessorNumber());
        if (closure_status[0] == 0) {
            has_python_pressure_closure_ = closure_status[1] != 0;
            has_python_ion_closure_ = closure_status[2] != 0;
        } else if (closure_status[0] == 1) {
            throw std::runtime_error("Embedded Python pressure closure setup failed for '" + pressure_namelist + "'");
        }
    }

    if (has_python_pressure_closure_) {
        define_pressure_gather_buffers(pressure_gather_);
    }
    if (has_python_ion_closure_) {
        define_pressure_gather_buffers(pressure_gather_i_);
    }
    print_closure_choice();
    print_dt_startup();

    update_pressure(level_.state, 0.0_rt, cfg_.dt, 0);
    if (cfg_.ions) {
        update_ion_pressure(level_.state, 0.0_rt, cfg_.dt, 0);
    }

    if (ParallelDescriptor::IOProcessor()) {
        Print() << "Initialized staggered single-level AMReX solver with AMReX FFT Helmholtz inversion\n";
    }
}

Simulation::~Simulation() = default;

void Simulation::define_level(const Geometry& geom, const BoxArray& ba) {
    level_.geom = geom;
    level_.ba = ba;
    level_.dmap = DistributionMapping(ba);

    vecops::define_field(level_.ba, level_.dmap, ng_, level_.state.b);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.state.q);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.state.u);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.state.p);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.stage.rhs);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.stage.rk_q0);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.stage.rk_accum);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.stage.work_b);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.stage.work_q);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.stage.work_cross);
    vecops::define_field(level_.ba, level_.dmap, ng_, level_.stage.work_divp);

    if (cfg_.ions) {
        vecops::define_field(level_.ba, level_.dmap, ng_, level_.state.q_i);
        vecops::define_field(level_.ba, level_.dmap, ng_, level_.state.u_i);
        vecops::define_field(level_.ba, level_.dmap, ng_, level_.state.p_i);
        vecops::define_field(level_.ba, level_.dmap, ng_, level_.stage.rhs_i);
        vecops::define_field(level_.ba, level_.dmap, ng_, level_.stage.rk_q0_i);
        vecops::define_field(level_.ba, level_.dmap, ng_, level_.stage.rk_accum_i);
    }
}

// One species' gather buffers: whole-domain copies on the IO rank, which the Python closure sees as NumPy arrays.
void Simulation::define_pressure_gather_buffers(GatheredPressureState& gather) {
    gather.ba = BoxArray(level_.geom.Domain());
    Vector<int> pmap(1, ParallelDescriptor::IOProcessorNumber());
    gather.dmap = DistributionMapping(pmap);

    auto define_one = [&](MultiFab& mf) {
        mf.define(gather.ba, gather.dmap, 1, 0);
        mf.setVal(0.0);
    };

    for (auto& mf : gather.b) {
        define_one(mf);
    }
    for (auto& mf : gather.q) {
        define_one(mf);
    }
    for (auto& mf : gather.u) {
        define_one(mf);
    }
    for (auto& mf : gather.p_in) {
        define_one(mf);
    }
    for (auto& mf : gather.p_out) {
        define_one(mf);
    }

    gather.defined = true;
}

// One startup line per species: which closure it uses, and with which coefficient.
void Simulation::print_closure_choice() const {
    if (!ParallelDescriptor::IOProcessor()) {
        return;
    }
    if (has_python_pressure_closure_) {
        Print() << std::setprecision(6)
                << "[closure] electrons: Python pressure_closure from the config; its coefficient is set"
                << " inside the closure (config nu = " << cfg_.nu << ")\n";
    } else {
        Print() << std::setprecision(6)
                << "[closure] electrons: built-in viscous stencil, coefficient nu = " << cfg_.nu << "\n";
    }
    if (!cfg_.ions) {
        return;
    }
    if (has_python_ion_closure_) {
        Print() << std::setprecision(6)
                << "[closure] ions: Python ion_pressure_closure from the config; its coefficient is set"
                << " inside the closure (config M = " << cfg_.mass_ratio << ", nu_i = " << cfg_.nu_i
                << ", M*nu_i = " << cfg_.mass_ratio * cfg_.nu_i << ")\n";
    } else {
        Print() << std::setprecision(6)
                << "[closure] ions: built-in viscous stencil, coefficient M*nu_i = " << cfg_.mass_ratio
                << " * " << cfg_.nu_i << " = " << cfg_.mass_ratio * cfg_.nu_i << "\n";
    }
}

void Simulation::initialize_state() {
    init_has_b_ = false;
    init_has_q_ = false;
    init_b_ghosts_ready_ = false;

    if (!cfg_.init_python_namelist.empty()) {
        initialize_state_from_python_namelist();
        return;
    }

    initialize_analytic_magnetic_field();
    init_has_b_ = true;
}

void Simulation::initialize_analytic_magnetic_field() {
    const Real two_pi = 2.0_rt * Math::pi<Real>();
    const Real b0 = cfg_.init_b0;
    const Real sigma = cfg_.init_sigma;
    const Real psi0 = cfg_.init_perturbation;
    const Real lx = cfg_.lx;
    const Real ly = cfg_.ly;
    const Real dx = level_.geom.CellSize(0);
    const Real dy = level_.geom.CellSize(1);
    const Real quarter_lx = Real(0.25) * lx;
    const Real quarter_ly = Real(0.25) * ly;

    for (MFIter mfi(level_.state.b.comp[vecops::X]); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto bx = level_.state.b.comp[vecops::X].array(mfi);
        const auto by = level_.state.b.comp[vecops::Y].array(mfi);
        const auto bz = level_.state.b.comp[vecops::Z].array(mfi);
        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
            const Real x_x = (i + Real(0.5)) * dx;
            const Real y_x = j * dy;
            const Real ph_x = (ly / (two_pi * sigma)) * std::cos(two_pi * y_x / ly);
            const Real sin_mode_x = std::sin(4.0_rt * two_pi * (y_x - quarter_ly) / ly);

            const Real x_y = i * dx;
            const Real y_y = (j + Real(0.5)) * dy;
            const Real cos_mode_y = std::cos(4.0_rt * two_pi * (y_y - quarter_ly) / ly);

            const Real y_z = (j + Real(0.5)) * dy;
            const Real ph_z = (ly / (two_pi * sigma)) * std::cos(two_pi * y_z / ly);

            bx(i, j, 0) = b0 * std::tanh(ph_x)
                         + 2.0_rt * (two_pi / ly) * psi0 *
                           std::cos(two_pi * (x_x - quarter_lx) / lx) * sin_mode_x;
            by(i, j, 0) = -psi0 * (two_pi / lx) *
                           std::sin(two_pi * (x_y - quarter_lx) / lx) * cos_mode_y;
            bz(i, j, 0) = b0 / std::cosh(ph_z);
        });
    }
}

void Simulation::initialize_state_from_python_namelist() {
    std::filesystem::path helper_path;
    for (const auto& candidate : {
             std::filesystem::path("scripts/generate_init_from_python.py"),
             std::filesystem::path(cfg_.config_dir) / ".." / "scripts" / "generate_init_from_python.py",
             std::filesystem::path(cfg_.config_dir) / "scripts" / "generate_init_from_python.py"}) {
        std::error_code ec;
        const auto canonical = std::filesystem::weakly_canonical(candidate, ec);
        if (!ec && std::filesystem::exists(canonical)) {
            helper_path = canonical;
            break;
        }
    }
    if (helper_path.empty()) {
        throw std::runtime_error("Missing Python initialization helper: scripts/generate_init_from_python.py");
    }

    const std::filesystem::path temp_root = make_shared_temp_path("qhd_init");

    if (ParallelDescriptor::IOProcessor()) {
        std::error_code ec;
        std::filesystem::remove_all(temp_root, ec);
        ec.clear();
        std::filesystem::create_directories(temp_root, ec);
        if (ec) {
            throw std::runtime_error("Failed to create temporary initialization directory '" +
                                     temp_root.string() + "': " + ec.message());
        }

        std::ostringstream cmd;
        const int active_nx = level_.geom.Domain().length(0);
        const int active_ny = level_.geom.Domain().length(1);
        cmd << "python3 "
            << quote_shell_arg(helper_path.string())
            << " --namelist " << quote_shell_arg(cfg_.init_python_namelist)
            << " --nx " << active_nx
            << " --ny " << active_ny
            << " --lx " << std::setprecision(17) << cfg_.lx
            << " --ly " << std::setprecision(17) << cfg_.ly
            << " --staggered"
            << " --output-dir " << quote_shell_arg(temp_root.string());

        const int rc = decode_process_exit_code(std::system(cmd.str().c_str()));
        if (rc != 0) {
            throw std::runtime_error("Python initialization failed for '" + cfg_.init_python_namelist +
                                     "' with exit code " + std::to_string(rc));
        }
    }

    ParallelDescriptor::Barrier();

    const auto has_family = [&](const char* prefix) {
        return std::filesystem::exists(temp_root / (std::string(prefix) + "x.npy")) &&
               std::filesystem::exists(temp_root / (std::string(prefix) + "y.npy")) &&
               std::filesystem::exists(temp_root / (std::string(prefix) + "z.npy"));
    };

    init_has_b_ = has_family("b");
    init_has_q_ = has_family("q");
    if (!init_has_b_ && !init_has_q_) {
        throw std::runtime_error("Python namelist did not define a complete B or Q field family");
    }

    const int active_nx = level_.geom.Domain().length(0);
    const int active_ny = level_.geom.Domain().length(1);
    if (init_has_b_) {
        load_array_into_multifab((temp_root / "bx.npy").string(), level_.state.b.comp[vecops::X], active_ny, active_nx, 0, false);
        load_array_into_multifab((temp_root / "by.npy").string(), level_.state.b.comp[vecops::Y], active_ny, active_nx, 0, false);
        load_array_into_multifab((temp_root / "bz.npy").string(), level_.state.b.comp[vecops::Z], active_ny, active_nx, 0, false);
        if (!init_has_q_) {
            load_array_into_multifab((temp_root / "bx.npy").string(), level_.stage.work_b.comp[vecops::X], active_ny, active_nx, 0, true);
            load_array_into_multifab((temp_root / "by.npy").string(), level_.stage.work_b.comp[vecops::Y], active_ny, active_nx, 0, true);
            load_array_into_multifab((temp_root / "bz.npy").string(), level_.stage.work_b.comp[vecops::Z], active_ny, active_nx, 0, true);
            init_b_ghosts_ready_ = true;
        }
    }
    if (init_has_q_) {
        load_array_into_multifab((temp_root / "qx.npy").string(), level_.state.q.comp[vecops::X], active_ny, active_nx, 0, false);
        load_array_into_multifab((temp_root / "qy.npy").string(), level_.state.q.comp[vecops::Y], active_ny, active_nx, 0, false);
        load_array_into_multifab((temp_root / "qz.npy").string(), level_.state.q.comp[vecops::Z], active_ny, active_nx, 0, false);
    }

    ParallelDescriptor::Barrier();
    if (ParallelDescriptor::IOProcessor()) {
        std::error_code ec;
        std::filesystem::remove_all(temp_root, ec);
    }
    ParallelDescriptor::Barrier();
}

void Simulation::load_array_into_multifab(const std::string& path,
                                          MultiFab& mf,
                                          int expected_ny,
                                          int expected_nx,
                                          int component,
                                          bool include_ghosts) const {
    if (mf.local_size() == 0) {
        return;
    }

    const auto data = read_npy_2d(path, expected_ny, expected_nx);
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const Box& box = include_ghosts ? mfi.fabbox() : mfi.validbox();
        const auto arr = mf.array(mfi);
        const auto lo = amrex::lbound(box);
        const auto hi = amrex::ubound(box);
        for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                const int ii = ((i % expected_nx) + expected_nx) % expected_nx;
                const int jj = ((j % expected_ny) + expected_ny) % expected_ny;
                arr(i, j, component) = data[static_cast<std::size_t>(jj) * expected_nx + ii];
            }
        }
    }
}

void Simulation::fill_level_ghosts(VectorField& field) {
    vecops::fill_periodic(level_.geom, field);
}

void Simulation::fill_level_ghosts(PressureTensor& field) {
    vecops::fill_periodic(level_.geom, field);
}

void Simulation::compute_u_from_b(const VectorField& b, VectorField& u, StageData& stage) {
    vecops::copy(b, stage.work_b);
    fill_level_ghosts(stage.work_b);
    vecops::compute_u_from_b_filled(level_.geom, stage.work_b, u);
}

void Simulation::compute_q_from_b(const VectorField& b, VectorField& u, VectorField& q, StageData& stage) {
    if (init_b_ghosts_ready_) {
        vecops::compute_u_from_b_filled(level_.geom, stage.work_b, u);
        fill_level_ghosts(u);
        vecops::compute_q_from_b_filled(level_.geom, stage.work_b, q);
        init_b_ghosts_ready_ = false;
        return;
    }

    vecops::copy(b, stage.work_b);
    fill_level_ghosts(stage.work_b);
    vecops::compute_u_from_b_filled(level_.geom, stage.work_b, u);
    fill_level_ghosts(u);
    vecops::compute_q_from_b_filled(level_.geom, stage.work_b, q);
}

// The built-in viscous stress P = -nu (grad u + grad u^T), the C++ twin of the shipped Python closure.
// It is the fallback for a species whose config has no Python closure: coefficient nu for the
// electrons, M*nu_i for the ions. nu is a parameter for that reason.
void Simulation::compute_default_pressure_from_filled(const VectorField& u, PressureTensor& p, Real nu) {
    const Real idx = Real(1.0) / level_.geom.CellSize(0);
    const Real idy = Real(1.0) / level_.geom.CellSize(1);

    if (nu == Real(0.0)) {
        vecops::set_val(p, 0.0);
        return;
    }

    for (MFIter mfi(p.comp[vecops::PXX]); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto ux = u.comp[vecops::X].const_array(mfi);
        const auto uy = u.comp[vecops::Y].const_array(mfi);
        const auto uz = u.comp[vecops::Z].const_array(mfi);
        const auto pxx = p.comp[vecops::PXX].array(mfi);
        const auto pxy = p.comp[vecops::PXY].array(mfi);
        const auto pxz = p.comp[vecops::PXZ].array(mfi);
        const auto pyy = p.comp[vecops::PYY].array(mfi);
        const auto pyz = p.comp[vecops::PYZ].array(mfi);
        const auto pzz = p.comp[vecops::PZZ].array(mfi);
        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
            pxx(i, j, 0) = -2.0_rt * nu * (ux(i, j, 0) - ux(i - 1, j, 0)) * idx;
            pxy(i, j, 0) = -nu * ((ux(i, j + 1, 0) - ux(i, j, 0)) * idy +
                                  (uy(i + 1, j, 0) - uy(i, j, 0)) * idx);
            pxz(i, j, 0) = -nu * (uz(i, j, 0) - uz(i - 1, j, 0)) * idx;
            pyy(i, j, 0) = -2.0_rt * nu * (uy(i, j, 0) - uy(i, j - 1, 0)) * idy;
            pyz(i, j, 0) = -nu * (uz(i, j, 0) - uz(i, j - 1, 0)) * idy;
            pzz(i, j, 0) = 0.0_rt;
        });
    }
}

// Electron stress P_e: the config's pressure_closure if it has one, else the built-in stencil with nu.
void Simulation::update_pressure(StateData& state, Real time, Real dt, int step) {
    if (!has_python_pressure_closure_) {
        compute_default_pressure_from_filled(state.u, state.p, cfg_.nu);
    } else {
        evaluate_python_closure(Species::electron, state.b, state.q, state.u, state.p, pressure_gather_,
                                time, dt, step);
    }
    fill_level_ghosts(state.p);
    ensure_finite_tensor("Pressure", state.p);
}

// Ion stress P_i, by the same rule as the electrons: the config's
// ion_pressure_closure if it has one, else the built-in stencil with coefficient M*nu_i (the
// M is written out, kept separate from nu_i). The ion closure sees Q_i, u_i, P_i under the names
// q, u, p. Needs u_i ghosts filled (refresh_derived_state_timed does that before calling this).
void Simulation::update_ion_pressure(StateData& state, Real time, Real dt, int step) {
    if (!has_python_ion_closure_) {
        compute_default_pressure_from_filled(state.u_i, state.p_i, cfg_.mass_ratio * cfg_.nu_i);
    } else {
        evaluate_python_closure(Species::ion, state.b, state.q_i, state.u_i, state.p_i, pressure_gather_i_,
                                time, dt, step);
    }
    fill_level_ghosts(state.p_i);
    ensure_finite_tensor("Ion pressure", state.p_i);
}

// Gather one species' fields to the IO rank, call that species' Python closure there, and copy the
// stress it wrote back into p. Only the fields the closure declared as dependencies are gathered.
void Simulation::evaluate_python_closure(Species species,
                                         const VectorField& b,
                                         const VectorField& q,
                                         const VectorField& u,
                                         PressureTensor& p,
                                         GatheredPressureState& gather,
                                         Real time,
                                         Real dt,
                                         int step) {
    AMREX_ALWAYS_ASSERT(gather.defined);

    if (pressure_engine_->needs_b(species)) {
        for (int n = 0; n < 3; ++n) {
            gather.b[n].ParallelCopy(b.comp[n], 0, 0, 1, 0, 0, level_.geom.periodicity());
        }
    }
    if (pressure_engine_->needs_q(species)) {
        for (int n = 0; n < 3; ++n) {
            gather.q[n].ParallelCopy(q.comp[n], 0, 0, 1, 0, 0, level_.geom.periodicity());
        }
    }
    if (pressure_engine_->needs_u(species)) {
        for (int n = 0; n < 3; ++n) {
            gather.u[n].ParallelCopy(u.comp[n], 0, 0, 1, 0, 0, level_.geom.periodicity());
        }
    }
    if (pressure_engine_->needs_p(species)) {
        for (int n = 0; n < 6; ++n) {
            gather.p_in[n].ParallelCopy(p.comp[n], 0, 0, 1, 0, 0, level_.geom.periodicity());
        }
    }

    int io_status = 0;
    if (ParallelDescriptor::IOProcessor()) {
        try {
            pressure_engine_->evaluate(species, time, dt, step,
                                       gather.b, gather.q, gather.u, gather.p_in, gather.p_out);
        } catch (const std::exception& ex) {
            Print() << "[closure] Python error: " << ex.what() << "\n";
            io_status = 1;
        } catch (...) {
            io_status = 1;
        }
    }

    ParallelDescriptor::Bcast(&io_status, 1, ParallelDescriptor::IOProcessorNumber());
    if (io_status != 0) {
        throw std::runtime_error(species == Species::ion
                                     ? "Embedded Python ion pressure closure evaluation failed"
                                     : "Embedded Python pressure closure evaluation failed");
    }
    for (int n = 0; n < 6; ++n) {
        p.comp[n].ParallelCopy(gather.p_out[n], 0, 0, 1, 0, 0, level_.geom.periodicity());
    }
}

void Simulation::compute_div_pressure_from_filled(const PressureTensor& p, VectorField& divp) {
    const Real idx = Real(1.0) / level_.geom.CellSize(0);
    const Real idy = Real(1.0) / level_.geom.CellSize(1);

    for (MFIter mfi(divp.comp[vecops::X]); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto pxx = p.comp[vecops::PXX].const_array(mfi);
        const auto pxy = p.comp[vecops::PXY].const_array(mfi);
        const auto pxz = p.comp[vecops::PXZ].const_array(mfi);
        const auto pyy = p.comp[vecops::PYY].const_array(mfi);
        const auto pyz = p.comp[vecops::PYZ].const_array(mfi);
        const auto divpx = divp.comp[vecops::X].array(mfi);
        const auto divpy = divp.comp[vecops::Y].array(mfi);
        const auto divpz = divp.comp[vecops::Z].array(mfi);
        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
            divpx(i, j, 0) = (pxx(i + 1, j, 0) - pxx(i, j, 0)) * idx +
                             (pxy(i, j, 0) - pxy(i, j - 1, 0)) * idy;
            divpy(i, j, 0) = (pxy(i, j, 0) - pxy(i - 1, j, 0)) * idx +
                             (pyy(i, j + 1, 0) - pyy(i, j, 0)) * idy;
            divpz(i, j, 0) = (pxz(i + 1, j, 0) - pxz(i, j, 0)) * idx +
                             (pyz(i, j + 1, 0) - pyz(i, j, 0)) * idy;
        });
    }
}

// Step 4: dQ_e/dt into stage.rhs and, with ions on, dQ_i/dt into stage.rhs_i.
void Simulation::compute_rhs(const StateData& state, StageData& stage) {
    compute_species_rhs(state.u, state.q, state.p, state.b, cfg_.eta, stage.rhs, stage);
    if (cfg_.ions) {
        // No resistive term with ions on: the two-species model is collisionless for now, and
        // Config.cpp enforces eta = 0.
        compute_species_rhs(state.u_i, state.q_i, state.p_i, state.b, 0.0_rt, stage.rhs_i, stage);
    }
}

// One species: rhs = curl(u x Q - div P), then the electron-only resistive term -eta*(Q + B).
void Simulation::compute_species_rhs(const VectorField& u,
                                     const VectorField& q,
                                     const PressureTensor& p,
                                     const VectorField& b,
                                     Real eta,
                                     VectorField& rhs,
                                     StageData& stage) {
    vecops::copy(b, stage.work_b);
    vecops::copy(q, stage.work_q);
    fill_level_ghosts(stage.work_b);
    fill_level_ghosts(stage.work_q);

    vecops::compute_cross_product(u, stage.work_q, stage.work_cross);
    compute_div_pressure_from_filled(p, stage.work_divp);
    vecops::saxpy(stage.work_cross, -1.0, stage.work_divp);
    fill_level_ghosts(stage.work_cross);
    vecops::compute_curl_from_filled(level_.geom, stage.work_cross, rhs);

    if (eta != Real(0.0)) {
        for (int n = 0; n < 3; ++n) {
            for (MFIter mfi(rhs.comp[n]); mfi.isValid(); ++mfi) {
                const Box& box = mfi.validbox();
                const auto out = rhs.comp[n].array(mfi);
                const auto qarr = stage.work_q.comp[n].const_array(mfi);
                const auto barr = stage.work_b.comp[n].const_array(mfi);
                ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                    out(i, j, 0) -= eta * (qarr(i, j, 0) + barr(i, j, 0));
                });
            }
        }
    }

    ensure_finite_field("RHS", rhs);
}

void Simulation::refresh_derived_state(StateData& state,
                                       StageData& stage,
                                       Real time,
                                       Real dt,
                                       int step) {
    // One copy of the refresh sequence (B -> u -> P), so the ion branches live in one place.
    Real helmholtz_wall = 0.0;
    Real velocity_wall = 0.0;
    Real pressure_wall = 0.0;
    refresh_derived_state_timed(state, stage, time, dt, step, helmholtz_wall, velocity_wall, pressure_wall);
}

void Simulation::refresh_derived_state_timed(StateData& state,
                                             StageData& stage,
                                             Real time,
                                             Real dt,
                                             int step,
                                             Real& helmholtz_wall_total,
                                             Real& velocity_wall_total,
                                             Real& pressure_wall_total) {
    const Real helmholtz_wall_start = ParallelDescriptor::second();
    if (cfg_.ions) {
        solve_helmholtz_ions(state.q, state.q_i, state.b, stage);
    } else {
        solve_helmholtz(state.q, state.b);
    }
    helmholtz_wall_total += ParallelDescriptor::second() - helmholtz_wall_start;

    const Real velocity_wall_start = ParallelDescriptor::second();
    if (cfg_.ions) {
        compute_ion_velocity(state, stage);
    }
    compute_u_from_b(state.b, state.u, stage);
    if (cfg_.ions) {
        // Step 3: u_e = u_i - curl B (compute_u_from_b gave -curl B).
        vecops::saxpy(state.u, Real(1.0), state.u_i);
    }
    fill_level_ghosts(state.u);
    velocity_wall_total += ParallelDescriptor::second() - velocity_wall_start;

    const Real pressure_wall_start = ParallelDescriptor::second();
    update_pressure(state, time, dt, step);
    if (cfg_.ions) {
        update_ion_pressure(state, time, dt, step);
    }
    pressure_wall_total += ParallelDescriptor::second() - pressure_wall_start;
}

void Simulation::evaluate_rhs_stage(StateData& state,
                                    StageData& stage,
                                    Real time,
                                    Real dt,
                                    int step,
                                    Real& rhs_wall_total,
                                    Real& helmholtz_wall_total,
                                    Real& velocity_wall_total,
                                    Real& pressure_wall_total) {
    refresh_derived_state_timed(state, stage, time, dt, step,
                                helmholtz_wall_total, velocity_wall_total, pressure_wall_total);

    const Real rhs_wall_start = ParallelDescriptor::second();
    compute_rhs(state, stage);
    rhs_wall_total += ParallelDescriptor::second() - rhs_wall_start;
}

void Simulation::advance_rk4(Real time,
                             Real dt,
                             int step,
                             Real& rhs_wall_total,
                             Real& helmholtz_wall_total,
                             Real& velocity_wall_total,
                             Real& pressure_wall_total) {
    auto& state = level_.state;
    auto& stage = level_.stage;
    const int next_step = step + 1;
    const bool ions = cfg_.ions != 0;

    // Ion twins of the Q_e operations below: same RK4 weights applied to Q_i; no-ops with ions off.
    auto ion_save_q0 = [&]() {
        if (ions) {
            vecops::copy(state.q_i, stage.rk_q0_i);
            vecops::set_val(stage.rk_accum_i, 0.0_rt);
        }
    };
    auto ion_accumulate = [&](Real weight) {
        if (ions) {
            vecops::saxpy(stage.rk_accum_i, weight, stage.rhs_i);
        }
    };
    auto ion_set_q = [&](Real h, const VectorField& increment) {
        if (ions) {
            vecops::copy(stage.rk_q0_i, state.q_i);
            vecops::saxpy(state.q_i, h, increment);
        }
    };

    vecops::copy(state.q, stage.rk_q0);
    vecops::set_val(stage.rk_accum, 0.0_rt);
    ion_save_q0();

    const Real rhs_wall_start = ParallelDescriptor::second();
    compute_rhs(state, stage);
    rhs_wall_total += ParallelDescriptor::second() - rhs_wall_start;
    vecops::saxpy(stage.rk_accum, 1.0_rt / 6.0_rt, stage.rhs);
    ion_accumulate(1.0_rt / 6.0_rt);

    vecops::copy(stage.rk_q0, state.q);
    vecops::saxpy(state.q, 0.5_rt * dt, stage.rhs);
    ion_set_q(0.5_rt * dt, stage.rhs_i);
    evaluate_rhs_stage(state, stage, time + 0.5_rt * dt, dt, next_step,
                       rhs_wall_total, helmholtz_wall_total, velocity_wall_total, pressure_wall_total);
    vecops::saxpy(stage.rk_accum, 1.0_rt / 3.0_rt, stage.rhs);
    ion_accumulate(1.0_rt / 3.0_rt);

    vecops::copy(stage.rk_q0, state.q);
    vecops::saxpy(state.q, 0.5_rt * dt, stage.rhs);
    ion_set_q(0.5_rt * dt, stage.rhs_i);
    evaluate_rhs_stage(state, stage, time + 0.5_rt * dt, dt, next_step,
                       rhs_wall_total, helmholtz_wall_total, velocity_wall_total, pressure_wall_total);
    vecops::saxpy(stage.rk_accum, 1.0_rt / 3.0_rt, stage.rhs);
    ion_accumulate(1.0_rt / 3.0_rt);

    vecops::copy(stage.rk_q0, state.q);
    vecops::saxpy(state.q, dt, stage.rhs);
    ion_set_q(dt, stage.rhs_i);
    evaluate_rhs_stage(state, stage, time + dt, dt, next_step,
                       rhs_wall_total, helmholtz_wall_total, velocity_wall_total, pressure_wall_total);
    vecops::saxpy(stage.rk_accum, 1.0_rt / 6.0_rt, stage.rhs);
    ion_accumulate(1.0_rt / 6.0_rt);

    vecops::copy(stage.rk_q0, state.q);
    vecops::saxpy(state.q, dt, stage.rk_accum);
    ion_set_q(dt, stage.rk_accum_i);
    refresh_derived_state_timed(state, stage, time + dt, dt, next_step,
                                helmholtz_wall_total, velocity_wall_total, pressure_wall_total);
}

void Simulation::solve_helmholtz(const VectorField& q, VectorField& b) {
    const int nx = level_.geom.Domain().length(0);
    const int ny = level_.geom.Domain().length(1);
    const Real idx2 = Real(1.0) / (level_.geom.CellSize(0) * level_.geom.CellSize(0));
    const Real idy2 = Real(1.0) / (level_.geom.CellSize(1) * level_.geom.CellSize(1));
    const Real two_pi = Real(2.0) * Math::pi<Real>();
    const Real scaling = helmholtz_fft_->scalingFactor();

    for (int comp = 0; comp < 3; ++comp) {
        MultiFab& bcomp = b.comp[comp];
        const MultiFab& qcomp = q.comp[comp];
        ensure_finite("Helmholtz q", qcomp);

        helmholtz_fft_->forwardThenBackward(
            qcomp, bcomp,
            [=] AMREX_GPU_DEVICE(int kx, int ky, int, auto& sp) noexcept {
                const Real theta_x = two_pi * static_cast<Real>(kx) / static_cast<Real>(nx);
                const Real theta_y = two_pi * static_cast<Real>(ky) / static_cast<Real>(ny);
                const Real lambda_x = Real(2.0) * (std::cos(theta_x) - Real(1.0)) * idx2;
                const Real lambda_y = Real(2.0) * (std::cos(theta_y) - Real(1.0)) * idy2;
                const Real denom = (lambda_x + lambda_y) - Real(1.0);
                sp *= scaling / denom;
            });
        ensure_finite("Helmholtz b", bcomp);
    }
}

// Step 2 with ions: lap B - (1 + 1/M) B = Q_e - Q_i/M. As M -> infinity this is solve_helmholtz.
void Simulation::solve_helmholtz_ions(const VectorField& q_e,
                                      const VectorField& q_i,
                                      VectorField& b,
                                      StageData& stage) {
    const int nx = level_.geom.Domain().length(0);
    const int ny = level_.geom.Domain().length(1);
    const Real idx2 = Real(1.0) / (level_.geom.CellSize(0) * level_.geom.CellSize(0));
    const Real idy2 = Real(1.0) / (level_.geom.CellSize(1) * level_.geom.CellSize(1));
    const Real two_pi = Real(2.0) * Math::pi<Real>();
    const Real scaling = helmholtz_fft_->scalingFactor();
    const Real inv_m = Real(1.0) / cfg_.mass_ratio;

    // Source Q_e - Q_i/M into stage.work_q (valid cells; the FFT needs no ghosts).
    vecops::lincomb(stage.work_q, Real(1.0), q_e, -inv_m, q_i);

    for (int comp = 0; comp < 3; ++comp) {
        MultiFab& bcomp = b.comp[comp];
        const MultiFab& src = stage.work_q.comp[comp];
        ensure_finite("Helmholtz source (ions)", src);

        helmholtz_fft_->forwardThenBackward(
            src, bcomp,
            [=] AMREX_GPU_DEVICE(int kx, int ky, int, auto& sp) noexcept {
                const Real theta_x = two_pi * static_cast<Real>(kx) / static_cast<Real>(nx);
                const Real theta_y = two_pi * static_cast<Real>(ky) / static_cast<Real>(ny);
                const Real lambda_x = Real(2.0) * (std::cos(theta_x) - Real(1.0)) * idx2;
                const Real lambda_y = Real(2.0) * (std::cos(theta_y) - Real(1.0)) * idy2;
                const Real denom = (lambda_x + lambda_y) - (Real(1.0) + inv_m);
                sp *= scaling / denom;
            });
        ensure_finite("Helmholtz b", bcomp);
    }
}

// Step 3: omega_i = (Q_i - B)/M, then lap u_i = -curl omega_i solved by FFT (div u_i = 0).
// Needs the current B, so it runs after the B solve.
void Simulation::compute_ion_velocity(StateData& state, StageData& stage) {
    const int nx = level_.geom.Domain().length(0);
    const int ny = level_.geom.Domain().length(1);
    const Real idx2 = Real(1.0) / (level_.geom.CellSize(0) * level_.geom.CellSize(0));
    const Real idy2 = Real(1.0) / (level_.geom.CellSize(1) * level_.geom.CellSize(1));
    const Real two_pi = Real(2.0) * Math::pi<Real>();
    const Real scaling = helmholtz_fft_->scalingFactor();
    const Real inv_m = Real(1.0) / cfg_.mass_ratio;

    // omega_i into stage.work_q, ghosts filled for the curl stencil; s = curl omega_i into stage.work_cross.
    vecops::lincomb(stage.work_q, inv_m, state.q_i, -inv_m, state.b);
    fill_level_ghosts(stage.work_q);
    vecops::compute_curl_from_filled(level_.geom, stage.work_q, stage.work_cross);

    for (int comp = 0; comp < 3; ++comp) {
        MultiFab& ucomp = state.u_i.comp[comp];
        const MultiFab& scomp = stage.work_cross.comp[comp];
        ensure_finite("Ion velocity source", scomp);

        helmholtz_fft_->forwardThenBackward(
            scomp, ucomp,
            [=] AMREX_GPU_DEVICE(int kx, int ky, int, auto& sp) noexcept {
                const Real theta_x = two_pi * static_cast<Real>(kx) / static_cast<Real>(nx);
                const Real theta_y = two_pi * static_cast<Real>(ky) / static_cast<Real>(ny);
                const Real lambda_x = Real(2.0) * (std::cos(theta_x) - Real(1.0)) * idx2;
                const Real lambda_y = Real(2.0) * (std::cos(theta_y) - Real(1.0)) * idy2;
                if (kx == 0 && ky == 0) {
                    // k = 0 ion velocity set to zero (plasma at rest on average): Q_i carries no information
                    // about a uniform drift.
                    sp = Real(0.0);
                } else {
                    sp *= -scaling / (lambda_x + lambda_y);
                }
            });
        ensure_finite("Ion velocity", ucomp);
    }
    fill_level_ghosts(state.u_i);
}

void Simulation::write_output(int step, Real time) {
    Vector<std::string> names = {
        "Bx", "By", "Bz", "Ux", "Uy", "Uz", "Qx", "Qy", "Qz",
        "Pxx", "Pxy", "Pxz", "Pyy", "Pyz", "Pzz", "TrP"};
    if (cfg_.ions) {
        for (const char* name : {"Qix", "Qiy", "Qiz", "Uix", "Uiy", "Uiz"}) {
            names.push_back(name);
        }
    }

    fill_level_ghosts(level_.state.b);
    fill_level_ghosts(level_.state.u);
    fill_level_ghosts(level_.state.q);
    fill_level_ghosts(level_.state.p);

    MultiFab plot_data(level_.ba, level_.dmap, static_cast<int>(names.size()), 0);
    for (MFIter mfi(plot_data); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto bx = level_.state.b.comp[vecops::X].const_array(mfi);
        const auto by = level_.state.b.comp[vecops::Y].const_array(mfi);
        const auto bz = level_.state.b.comp[vecops::Z].const_array(mfi);
        const auto ux = level_.state.u.comp[vecops::X].const_array(mfi);
        const auto uy = level_.state.u.comp[vecops::Y].const_array(mfi);
        const auto uz = level_.state.u.comp[vecops::Z].const_array(mfi);
        const auto qx = level_.state.q.comp[vecops::X].const_array(mfi);
        const auto qy = level_.state.q.comp[vecops::Y].const_array(mfi);
        const auto qz = level_.state.q.comp[vecops::Z].const_array(mfi);
        const auto pxx = level_.state.p.comp[vecops::PXX].const_array(mfi);
        const auto pxy = level_.state.p.comp[vecops::PXY].const_array(mfi);
        const auto pxz = level_.state.p.comp[vecops::PXZ].const_array(mfi);
        const auto pyy = level_.state.p.comp[vecops::PYY].const_array(mfi);
        const auto pyz = level_.state.p.comp[vecops::PYZ].const_array(mfi);
        const auto pzz = level_.state.p.comp[vecops::PZZ].const_array(mfi);
        const auto out = plot_data.array(mfi);
        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
            out(i, j, 0) = 0.5_rt * (bx(i, j, 0) + bx(i, j + 1, 0));
            out(i, j, 1) = 0.5_rt * (by(i, j, 0) + by(i + 1, j, 0));
            out(i, j, 2) = bz(i, j, 0);

            out(i, j, 3) = 0.5_rt * (ux(i, j, 0) + ux(i, j + 1, 0));
            out(i, j, 4) = 0.5_rt * (uy(i, j, 0) + uy(i + 1, j, 0));
            out(i, j, 5) = uz(i, j, 0);

            out(i, j, 6) = 0.5_rt * (qx(i, j, 0) + qx(i, j + 1, 0));
            out(i, j, 7) = 0.5_rt * (qy(i, j, 0) + qy(i + 1, j, 0));
            out(i, j, 8) = qz(i, j, 0);

            out(i, j, 9) = 0.25_rt * (pxx(i, j, 0) + pxx(i + 1, j, 0) +
                                      pxx(i, j + 1, 0) + pxx(i + 1, j + 1, 0));
            out(i, j, 10) = pxy(i, j, 0);
            out(i, j, 11) = 0.5_rt * (pxz(i, j, 0) + pxz(i + 1, j, 0));
            out(i, j, 12) = 0.25_rt * (pyy(i, j, 0) + pyy(i + 1, j, 0) +
                                       pyy(i, j + 1, 0) + pyy(i + 1, j + 1, 0));
            out(i, j, 13) = 0.5_rt * (pyz(i, j, 0) + pyz(i, j + 1, 0));
            out(i, j, 14) = pzz(i, j, 0);
            out(i, j, 15) = out(i, j, 9) + out(i, j, 12) + out(i, j, 14);
        });
    }

    if (cfg_.ions) {
        // Ion fields after TrP, averaged to cell centres exactly like Q and U above.
        fill_level_ghosts(level_.state.q_i);
        fill_level_ghosts(level_.state.u_i);
        for (MFIter mfi(plot_data); mfi.isValid(); ++mfi) {
            const Box& box = mfi.validbox();
            const auto qix = level_.state.q_i.comp[vecops::X].const_array(mfi);
            const auto qiy = level_.state.q_i.comp[vecops::Y].const_array(mfi);
            const auto qiz = level_.state.q_i.comp[vecops::Z].const_array(mfi);
            const auto uix = level_.state.u_i.comp[vecops::X].const_array(mfi);
            const auto uiy = level_.state.u_i.comp[vecops::Y].const_array(mfi);
            const auto uiz = level_.state.u_i.comp[vecops::Z].const_array(mfi);
            const auto out = plot_data.array(mfi);
            ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                out(i, j, 16) = 0.5_rt * (qix(i, j, 0) + qix(i, j + 1, 0));
                out(i, j, 17) = 0.5_rt * (qiy(i, j, 0) + qiy(i + 1, j, 0));
                out(i, j, 18) = qiz(i, j, 0);

                out(i, j, 19) = 0.5_rt * (uix(i, j, 0) + uix(i, j + 1, 0));
                out(i, j, 20) = 0.5_rt * (uiy(i, j, 0) + uiy(i + 1, j, 0));
                out(i, j, 21) = uiz(i, j, 0);
            });
        }
    }

    std::ostringstream oss;
    oss << cfg_.output_dir << "/plt_" << std::setw(6) << std::setfill('0') << step;
#ifdef AMREX_USE_HDF5
    WriteSingleLevelPlotfileHDF5(oss.str(), plot_data, names, level_.geom, time, step);
#else
    WriteSingleLevelPlotfile(oss.str(), plot_data, names, level_.geom, time, step);
#endif
}

// One grep-able "[ions]" line per output. Divergences use the node stencil that matches the
// staggered layout: (fx(i) - fx(i-1))/dx + (fy(j) - fy(j-1))/dy at (i, j); div(curl) = 0 exactly with it.
void Simulation::print_ion_diagnostics(int step, Real time) {
    auto& state = level_.state;
    auto& stage = level_.stage;
    const Real idx = Real(1.0) / level_.geom.CellSize(0);
    const Real idy = Real(1.0) / level_.geom.CellSize(1);
    const Real inv_m = Real(1.0) / cfg_.mass_ratio;
    const Real npts = static_cast<Real>(level_.geom.Domain().numPts());

    auto max_abs_div = [&](VectorField& f) {
        fill_level_ghosts(f);
        MultiFab div(level_.ba, level_.dmap, 1, 0);
        for (MFIter mfi(div); mfi.isValid(); ++mfi) {
            const Box& box = mfi.validbox();
            const auto fx = f.comp[vecops::X].const_array(mfi);
            const auto fy = f.comp[vecops::Y].const_array(mfi);
            const auto d = div.array(mfi);
            ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
                d(i, j, 0) = (fx(i, j, 0) - fx(i - 1, j, 0)) * idx + (fy(i, j, 0) - fy(i, j - 1, 0)) * idy;
            });
        }
        return div.norm0();
    };
    auto max_abs = [](const VectorField& f) {
        return std::max({f.comp[0].norm0(), f.comp[1].norm0(), f.comp[2].norm0()});
    };

    // omega_i into work_q; curl u_i - omega_i into work_cross (both are free between steps).
    vecops::lincomb(stage.work_q, inv_m, state.q_i, -inv_m, state.b);
    fill_level_ghosts(state.u_i);
    vecops::compute_curl_from_filled(level_.geom, state.u_i, stage.work_cross);
    vecops::saxpy(stage.work_cross, -1.0_rt, stage.work_q);

    const Real div_ui = max_abs_div(state.u_i);
    const Real div_b = max_abs_div(state.b);
    const Real div_qi = max_abs_div(state.q_i);
    const Real curl_residual = max_abs(stage.work_cross);
    const Real omega_max = max_abs(stage.work_q);

    if (ParallelDescriptor::IOProcessor()) {
        Print() << std::scientific << std::setprecision(3)
                << "[ions] step=" << step << " t=" << time
                << " max|div u_i|=" << div_ui
                << " max|curl u_i - omega_i|=" << curl_residual
                << " (max|omega_i|=" << omega_max << ")"
                << " max|div B|=" << div_b
                << " max|div Q_i|=" << div_qi
                << " <u_i>=(" << state.u_i.comp[0].sum() / npts << ", " << state.u_i.comp[1].sum() / npts
                << ", " << state.u_i.comp[2].sum() / npts << ")"
                << " <u_e>=(" << state.u.comp[0].sum() / npts << ", " << state.u.comp[1].sum() / npts
                << ", " << state.u.comp[2].sum() / npts << ")\n";
    }
}

// --- dt diagnostics ------------------------------------------------------------------------
// dt is an input set in the Python config; the solver has no dt rule. These
// prints only let the config's rule be checked against the RK4 stability limits. They read the
// state and change nothing.

// Advective number of one species: dt * max over cells of (|u_x|/dx + |u_y|/dy). u_x and u_y are
// taken as stored: u_x on the x-edge (i+1/2, j) and u_y on the y-edge (i, j+1/2) of the same
// (i, j), not averaged to one point. Compare with RK4's imaginary-axis limit 2*sqrt(2).
Real Simulation::advective_number(const VectorField& u, Real dt) const {
    const Real idx = Real(1.0) / level_.geom.CellSize(0);
    const Real idy = Real(1.0) / level_.geom.CellSize(1);
    MultiFab rate(level_.ba, level_.dmap, 1, 0);
    for (MFIter mfi(rate); mfi.isValid(); ++mfi) {
        const Box& box = mfi.validbox();
        const auto ux = u.comp[vecops::X].const_array(mfi);
        const auto uy = u.comp[vecops::Y].const_array(mfi);
        const auto r = rate.array(mfi);
        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept {
            r(i, j, 0) = std::abs(ux(i, j, 0)) * idx + std::abs(uy(i, j, 0)) * idy;
        });
    }
    return dt * rate.max(0);
}

// At startup: dt, and the leading-order viscous number dt * nu * |lambda|max per species, where
// |lambda|max = 4/dx^2 + 4/dy^2 is the largest eigenvalue of the 5-point Laplacian (checkerboard
// mode). Compare with RK4's real-axis limit 2.785. Leading order only: the exact two-species rate
// has Helmholtz corrections of relative size 1/|lambda|. For the ions both candidate coefficients
// are printed: M nu_i lap(omega_i) = nu_i lap(Q_i - B), so the rate on Q_i is nu_i |lambda| if
// the B term is negligible at large |lambda| [derived], and M nu_i |lambda| if it were not.
void Simulation::print_dt_startup() const {
    if (!ParallelDescriptor::IOProcessor()) {
        return;
    }
    const Real dx = level_.geom.CellSize(0);
    const Real dy = level_.geom.CellSize(1);
    const Real lam = Real(4.0) / (dx * dx) + Real(4.0) / (dy * dy);
    const Real dt = cfg_.dt;
    Print() << std::setprecision(6)
            << "[dt] dt = " << dt << " (set in the config; the solver has no dt rule)."
            << " RK4 limits: real axis 2.785, imaginary axis 2*sqrt(2) = 2.828\n"
            << "[dt] viscous number, leading order, dt*nu*|lambda|max with |lambda|max = 4/dx^2 + 4/dy^2 = "
            << lam << "; limit 2.785\n"
            << "[dt]   electrons  dt*nu_e*|lambda|max   = " << dt * cfg_.nu * lam << "\n";
    if (cfg_.ions) {
        Print() << std::setprecision(6)
                << "[dt]   ions       dt*nu_i*|lambda|max   = " << dt * cfg_.nu_i * lam
                << "  (if M cancels: M nu_i lap(omega_i) = nu_i lap(Q_i - B))\n"
                << "[dt]   ions       dt*M*nu_i*|lambda|max = " << dt * cfg_.mass_ratio * cfg_.nu_i * lam
                << "  (if it does not)\n";
    }
    Print() << "[dt] advective number at each output: adv = dt*max(|u_x|/dx + |u_y|/dy), u_x and u_y"
            << " as stored on their own staggered edges; limit 2.828\n";
}

// At each output: the advective number of each species with the config's dt.
void Simulation::print_dt_diagnostics(int step, Real time) const {
    const Real adv_e = advective_number(level_.state.u, cfg_.dt);
    const Real adv_i = cfg_.ions ? advective_number(level_.state.u_i, cfg_.dt) : Real(0.0);
    if (ParallelDescriptor::IOProcessor()) {
        std::ostringstream line;
        line << std::scientific << std::setprecision(4)
             << "[dt] step=" << step << " t=" << time << " adv_e=" << adv_e;
        if (cfg_.ions) {
            line << " adv_i=" << adv_i;
        }
        line << " (limit 2.828)\n";
        Print() << line.str();
    }
}

void Simulation::run() {
    Real time = 0.0;
    int step = 0;
    Real rhs_wall_total = 0.0;
    Real helmholtz_wall_total = 0.0;
    Real velocity_wall_total = 0.0;
    Real pressure_wall_total = 0.0;

    reset_output_dir(cfg_.output_dir);
    write_output(step, time);
    print_dt_diagnostics(step, time);
    if (cfg_.ions) {
        print_ion_diagnostics(step, time);
    }

    const Real start_wall = ParallelDescriptor::second();
    while (time < cfg_.t_end - Real(1.0e-14)) {
        const Real dt = std::min(cfg_.dt, cfg_.t_end - time);

        advance_rk4(time, dt, step,
                    rhs_wall_total, helmholtz_wall_total, velocity_wall_total, pressure_wall_total);

        time += dt;
        ++step;

        if (step % cfg_.output_every == 0 || time >= cfg_.t_end - Real(1.0e-14)) {
            write_output(step, time);
            if (ParallelDescriptor::IOProcessor()) {
                Print() << std::fixed << std::setprecision(6)
                        << "step=" << step
                        << " time=" << time
                        << " dt=" << dt
                        << " wall=" << (ParallelDescriptor::second() - start_wall) << " s\n";
            }
            print_dt_diagnostics(step, time);
            if (cfg_.ions) {
                print_ion_diagnostics(step, time);
            }
        }
    }

    if (ParallelDescriptor::IOProcessor()) {
        const Real total_wall = ParallelDescriptor::second() - start_wall;
        const Real accounted_wall = rhs_wall_total + helmholtz_wall_total + velocity_wall_total + pressure_wall_total;
        Print() << std::fixed << std::setprecision(6)
                << "[timing] rhs=" << rhs_wall_total << " s"
                << " (" << (100.0_rt * rhs_wall_total / std::max(total_wall, Real(1.0e-14))) << "%)"
                << " helmholtz=" << helmholtz_wall_total << " s"
                << " (" << (100.0_rt * helmholtz_wall_total / std::max(total_wall, Real(1.0e-14))) << "%)"
                << " velocity=" << velocity_wall_total << " s"
                << " (" << (100.0_rt * velocity_wall_total / std::max(total_wall, Real(1.0e-14))) << "%)"
                << " pressure=" << pressure_wall_total << " s"
                << " (" << (100.0_rt * pressure_wall_total / std::max(total_wall, Real(1.0e-14))) << "%)"
                << " accounted=" << accounted_wall << " s"
                << " total=" << total_wall << " s\n";
    }
}
