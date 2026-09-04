#!/usr/bin/env python3
"""Render an MP4 movie of out-of-plane current with in-plane B/Q field lines."""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass

# Import the shared movie helpers first; they configure Matplotlib for headless rendering.
from make_output_movie import (
    GridState,
    current_panel_cmap,
    field_extent,
    frame_title,
    iter_prepared_frames,
    prepare_frame,
    remove_contour,
    scalar_limits,
    select_steps,
)

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from read_output import discover_steps
from view_output_slider import (
    B_CONTOUR_COLOR,
    CANONICAL_CONTOUR_COLOR,
    compute_global_display_limits,
    safe_limits,
)


@dataclass
class CurrentFluxFigureState:
    title: object
    panel: GridState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output", help="Directory with AMReX plotfiles `plt_XXXXXX` or `plt_XXXXXX.h5`")
    parser.add_argument("--mp4", default=None, help="Output MP4 path (default: sibling of output dir)")
    parser.add_argument("--fps", type=float, default=10.0, help="Video frame rate")
    parser.add_argument("--start-step", type=int, default=None, help="First step to render")
    parser.add_argument("--end-step", type=int, default=None, help="Last step to render")
    parser.add_argument("--every", type=int, default=1, help="Render every Nth plotfile")
    parser.add_argument("--levels", type=int, default=16, help="Number of B/Q field-line contour levels")
    parser.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap for current background")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI for MP4 encoding")
    parser.add_argument(
        "--current-component",
        choices=["Jx", "Jy", "Jz", "Jmag"],
        default="Jz",
        help="Current-density quantity to show; default is the out-of-plane current Jz",
    )
    parser.add_argument(
        "--global-scale",
        action="store_true",
        help="Deprecated alias for the default robust global scaling behavior",
    )
    parser.add_argument(
        "--dynamic-scale",
        action="store_true",
        help="Recompute color and contour limits per frame instead of using one robust global scale",
    )
    parser.add_argument(
        "--outlier-percentile",
        type=float,
        default=1.0,
        help="Clip this percentile from each tail when computing global ranges (default: 1.0)",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=1,
        help="Worker threads used when scanning global display limits",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=1,
        help="Number of future frames to prepare in background while encoding (0 disables)",
    )
    parser.add_argument(
        "--ffmpeg-preset",
        default="medium",
        help="libx264 preset for ffmpeg; faster presets trade compression efficiency for speed",
    )
    return parser.parse_args()


def default_mp4_path(output_dir: pathlib.Path) -> pathlib.Path:
    resolved = output_dir.resolve()
    return resolved.parent / f"{resolved.name}_current_flux.mp4"


def field_line_legend_handles() -> list[object]:
    return [
        plt.Line2D([0], [0], color=B_CONTOUR_COLOR, lw=1.5, label="B field lines"),
        plt.Line2D([0], [0], color=CANONICAL_CONTOUR_COLOR, lw=1.5, label="Q field lines"),
    ]


def update_current_flux_panel(
    state: GridState,
    meta: dict,
    current: np.ndarray,
    current_label: str,
    x: np.ndarray,
    y: np.ndarray,
    b_psi: np.ndarray,
    q_psi: np.ndarray,
    levels: int,
    contour_limits: dict[str, tuple[float, float]] | None,
    current_limits: tuple[float, float] | None,
) -> None:
    prob_lo_x, prob_hi_x, prob_lo_y, prob_hi_y = field_extent(meta)
    current_vmin, current_vmax = scalar_limits(current, current_limits)

    state.image.set_data(current)
    state.image.set_extent((prob_lo_x, prob_hi_x, prob_lo_y, prob_hi_y))
    state.image.set_clim(current_vmin, current_vmax)

    if state.legend is not None:
        state.legend.remove()
    state.legend = state.ax.legend(handles=field_line_legend_handles(), loc="upper right", frameon=True, fontsize=8)

    remove_contour(state.b_contour)
    remove_contour(state.q_contour)
    if contour_limits is None:
        b_vmin, b_vmax = safe_limits(b_psi)
        q_vmin, q_vmax = safe_limits(q_psi)
    else:
        b_vmin, b_vmax = contour_limits["B"]
        q_vmin, q_vmax = contour_limits["Q"]

    state.b_contour = state.ax.contour(
        x,
        y,
        b_psi,
        levels=np.linspace(b_vmin, b_vmax, levels),
        linewidths=0.8,
        colors=B_CONTOUR_COLOR,
    )
    state.q_contour = state.ax.contour(
        x,
        y,
        q_psi,
        levels=np.linspace(q_vmin, q_vmax, levels),
        linewidths=0.8,
        colors=CANONICAL_CONTOUR_COLOR,
    )

    state.ax.set_title(f"{current_label} + B/Q field lines")
    state.ax.set_xlabel("x")
    state.ax.set_ylabel("y")
    state.ax.set_xlim(prob_lo_x, prob_hi_x)
    state.ax.set_ylim(prob_lo_y, prob_hi_y)
    state.ax.set_aspect("equal")
    state.ax.grid(False)
    state.colorbar.update_normal(state.image)


def build_current_flux_figure(
    fig: plt.Figure,
    frame,
    levels: int,
    cmap: str,
    current_component: str,
    contour_limits: dict[str, tuple[float, float]] | None,
    current_limits: tuple[float, float] | None,
) -> CurrentFluxFigureState:
    fig.clear()
    ax = fig.subplots()
    fig.subplots_adjust(left=0.11, right=0.91, top=0.9, bottom=0.12)

    current = frame.arrays["CurrentPanel"]
    current_vmin, current_vmax = scalar_limits(current, current_limits)
    image = ax.imshow(
        current,
        origin="lower",
        cmap=current_panel_cmap(current_component, cmap),
        interpolation="nearest",
        extent=field_extent(frame.meta),
        vmin=current_vmin,
        vmax=current_vmax,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(current_component)
    panel = GridState(ax=ax, image=image, colorbar=colorbar, legend=None)
    update_current_flux_panel(
        panel,
        frame.meta,
        current,
        current_component,
        frame.x,
        frame.y,
        frame.fluxes["B"],
        frame.fluxes["Q"],
        levels,
        contour_limits,
        current_limits,
    )

    title = fig.suptitle(frame_title(frame), fontsize=12)
    return CurrentFluxFigureState(title=title, panel=panel)


def update_current_flux_figure(
    state: CurrentFluxFigureState,
    frame,
    levels: int,
    current_component: str,
    contour_limits: dict[str, tuple[float, float]] | None,
    current_limits: tuple[float, float] | None,
) -> None:
    update_current_flux_panel(
        state.panel,
        frame.meta,
        frame.arrays["CurrentPanel"],
        current_component,
        frame.x,
        frame.y,
        frame.fluxes["B"],
        frame.fluxes["Q"],
        levels,
        contour_limits,
        current_limits,
    )
    state.title.set_text(frame_title(frame))


def main() -> int:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output directory does not exist: {output_dir}", file=sys.stderr)
        return 1
    if args.every <= 0:
        print("--every must be a positive integer", file=sys.stderr)
        return 1
    if args.levels < 2:
        print("--levels must be at least 2", file=sys.stderr)
        return 1
    if args.scan_workers < 1:
        print("--scan-workers must be at least 1", file=sys.stderr)
        return 1
    if args.prefetch < 0:
        print("--prefetch must be non-negative", file=sys.stderr)
        return 1

    all_steps = discover_steps(output_dir)
    if not all_steps:
        print(f"No plotfiles found in {output_dir}", file=sys.stderr)
        return 1

    steps = select_steps(all_steps, args.start_step, args.end_step, args.every)
    if not steps:
        print("No steps selected for movie generation", file=sys.stderr)
        return 1

    mp4_path = pathlib.Path(args.mp4) if args.mp4 is not None else default_mp4_path(output_dir)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    contour_limits = None
    current_limits = None
    if not args.dynamic_scale:
        _, contour_limits, current_limits = compute_global_display_limits(
            output_dir,
            steps,
            args.current_component,
            args.outlier_percentile,
            workers=min(args.scan_workers, len(steps)),
        )

    frame_iter = iter_prepared_frames(output_dir, steps, args.current_component, args.prefetch)
    try:
        first_frame = next(frame_iter)
    except StopIteration:
        print("No steps selected for movie generation", file=sys.stderr)
        return 1

    fig = plt.figure(figsize=(7.4, 6.4))
    state = build_current_flux_figure(
        fig,
        first_frame,
        args.levels,
        args.cmap,
        args.current_component,
        contour_limits,
        current_limits,
    )

    writer_kwargs = {"fps": args.fps, "codec": "libx264"}
    if args.ffmpeg_preset:
        writer_kwargs["extra_args"] = ["-preset", args.ffmpeg_preset]
    writer = FFMpegWriter(**writer_kwargs)

    print(
        f"[movie] writing {len(steps)} current/field-line frames from {output_dir} to {mp4_path} at {args.fps:g} fps",
        file=sys.stderr,
    )
    with writer.saving(fig, str(mp4_path), dpi=args.dpi):
        writer.grab_frame()
        if len(steps) == 1:
            print("[movie] rendered 1/1 frames", file=sys.stderr)
        elif len(steps) > 1:
            print(f"[movie] rendered 1/{len(steps)} frames", file=sys.stderr)

        for idx, frame in enumerate(frame_iter, start=2):
            update_current_flux_figure(
                state,
                frame,
                args.levels,
                args.current_component,
                contour_limits,
                current_limits,
            )
            writer.grab_frame()
            if idx == len(steps) or idx % 10 == 0:
                print(f"[movie] rendered {idx}/{len(steps)} frames", file=sys.stderr)

    plt.close(fig)
    print(f"[movie] wrote {mp4_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
