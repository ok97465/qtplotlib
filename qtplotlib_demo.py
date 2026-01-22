"""Run qtplotlib demos from a single script."""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from qtplotlib import imagescqt


@dataclass(frozen=True)
class DemoSpec:
    name: str
    summary: str
    runner: Callable[[argparse.Namespace], None]


def _make_imagesc_data(size: int, seed: int) -> np.ndarray:
    """Build sample data with structure + noise for visualization."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-3.2, 3.2, size)
    y = np.linspace(-2.4, 2.4, size)
    xx, yy = np.meshgrid(x, y)
    pattern = np.sin(xx * 2.2) * np.cos(yy * 3.1)
    ripple = 0.35 * np.cos(xx**2 + yy**2)
    noise = 0.15 * rng.standard_normal((size, size))
    return pattern + ripple + noise


def run_imagesc_demo(args: argparse.Namespace) -> None:
    """Launch an imagesc-style demo window."""
    size = args.size
    data = _make_imagesc_data(size, args.seed)
    x = np.linspace(-4.0, 4.0, size)
    y = np.linspace(-3.0, 3.0, size)
    window = imagescqt(
        x,
        y,
        data,
        cmap=args.cmap,
        aspect=args.aspect,
        title=args.title,
        xlabel="X axis",
        ylabel="Y axis",
        colorbar=args.colorbar,
        colorbar_label="Amplitude",
        interpolation=args.interp,
    )
    window.tight_layout(pad=1.04, auto_resize=True)


DEMO_SPECS: list[DemoSpec] = [
    DemoSpec(
        name="imagesc",
        summary="Imagesc-style 2D array viewer with axes, colorbar, and toolbar.",
        runner=run_imagesc_demo,
    ),
]
DEMO_INDEX = {spec.name: spec for spec in DEMO_SPECS}


def _print_demo_list() -> None:
    print("Available demos:")
    for spec in DEMO_SPECS:
        print(f"  - {spec.name}: {spec.summary}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="qtplotlib demo runner (add plot/hist demos here later)."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available demos.",
    )
    parser.add_argument(
        "--demo",
        default="imagesc",
        choices=sorted(DEMO_INDEX),
        help="Select which demo to run.",
    )
    parser.add_argument("--size", type=int, default=240, help="Square data size.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name.")
    parser.add_argument(
        "--aspect",
        default="equal",
        choices=["equal", "auto"],
        help="Aspect mode for the demo.",
    )
    parser.add_argument(
        "--interp",
        default="nearest",
        choices=["nearest", "bilinear"],
        help="Interpolation mode for the demo.",
    )
    parser.add_argument(
        "--colorbar",
        action="store_true",
        help="Enable the colorbar in the demo.",
    )
    parser.add_argument("--title", default="imagescqt demo", help="Window title.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for running demos from the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.list:
        _print_demo_list()
        return 0
    DEMO_INDEX[args.demo].runner(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
