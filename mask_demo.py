"""Run a grayscale imagescqt demo with colored mask overlays."""

import argparse
from collections.abc import Sequence

import numpy as np
from PySide6 import QtWidgets

from qtplotlib import imagescqt


def _make_demo_arrays(
    size: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a grayscale image and two binary masks."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-3.2, 3.2, size)
    y = np.linspace(-2.8, 2.8, size)
    xx, yy = np.meshgrid(x, y)

    image = (
        0.45 * np.sin(xx * 2.1)
        + 0.35 * np.cos(yy * 3.2)
        + 0.3 * np.sin(xx * yy)
        + 0.08 * rng.standard_normal((size, size))
    )
    lesion_mask = ((xx + 0.75) ** 2 / 0.75**2 + (yy - 0.25) ** 2 / 0.5**2) < 1.0
    ring = np.sqrt((xx - 0.85) ** 2 + (yy + 0.35) ** 2)
    boundary_mask = (ring > 0.62) & (ring < 0.78)
    return image, lesion_mask, boundary_mask


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="imagescqt mask overlay demo.")
    parser.add_argument("--size", type=int, default=768, help="Square data size.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the mask overlay demo."""
    args = _build_parser().parse_args(argv)
    image, lesion_mask, boundary_mask = _make_demo_arrays(args.size, args.seed)

    app = QtWidgets.QApplication.instance()
    created_app = app is None
    if app is None:
        app = QtWidgets.QApplication([])

    x = np.linspace(-3.2, 3.2, image.shape[1])
    y = np.linspace(-2.8, 2.8, image.shape[0])
    window = imagescqt(
        x,
        y,
        image,
        cmap="gray",
        title="imagescqt mask demo",
        xlabel="X axis",
        ylabel="Y axis",
        colorbar=True,
        colorbar_label="Intensity",
        mask=lesion_mask,
        mask_color="tab:red",
        mask_alpha=0.35,
        mask_label="lesion",
    )
    window.add_mask(
        boundary_mask,
        color="cyan",
        alpha=0.28,
        label="boundary",
    )
    window.tight_layout(pad=1.04, auto_resize=True)

    if created_app:
        return int(app.exec())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
