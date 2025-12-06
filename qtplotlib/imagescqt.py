"""Lightweight PySide6 viewer similar to MATLAB's imagesc."""

from math import hypot

import numpy as np
from matplotlib import cm
from numpy import asarray, clip, isfinite, linspace, log10, nan_to_num, ndarray, uint8

_COLORMAP_CACHE: dict[str, object] = {}

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:
    raise ImportError("PySide6 is required to use imagescqt.") from exc


def _normalize_data(
    data: ndarray, vmin: float | None, vmax: float | None
) -> tuple[ndarray, float, float]:
    """Normalize array into 0..1 with NaNs sent to middle gray, returning limits used."""
    vmin_resolved, vmax_resolved = _resolve_limits(data, vmin, vmax)
    norm = (data - vmin_resolved) / (vmax_resolved - vmin_resolved)
    norm = clip(norm, 0.0, 1.0)
    return nan_to_num(norm, nan=0.5), vmin_resolved, vmax_resolved


def _resolve_limits(
    data: ndarray, vmin: float | None, vmax: float | None
) -> tuple[float, float]:
    finite_vals = data[isfinite(data)]
    if finite_vals.size == 0:
        vmin = 0.0 if vmin is None else vmin
        vmax = 1.0 if vmax is None else vmax
    else:
        if vmin is None:
            vmin = float(finite_vals.min())
        if vmax is None:
            vmax = float(finite_vals.max())
        if vmin == vmax:
            vmax = vmin + 1e-9

    return float(vmin), float(vmax)


def _get_cmap(cmap: str):
    cmap_obj = _COLORMAP_CACHE.get(cmap)
    if cmap_obj is None:
        cmap_obj = cm.get_cmap(cmap)
        _COLORMAP_CACHE[cmap] = cmap_obj
    return cmap_obj


def _apply_colormap(data: ndarray, cmap: str) -> ndarray:
    """Convert normalized data into RGBA image."""
    rgba = _get_cmap(cmap)(data)
    rgba_uint8 = (clip(rgba, 0.0, 1.0) * 255).astype(uint8)
    return rgba_uint8


def _make_colorbar_image(cmap: str, height: int = 256) -> QtGui.QImage:
    """Create a 1-pixel wide vertical colorbar image (top=max, bottom=min)."""
    grad = linspace(1.0, 0.0, height, dtype=float).reshape(height, 1)
    rgba_uint8 = _apply_colormap(grad, cmap=cmap)
    image = QtGui.QImage(
        rgba_uint8.data,
        1,
        height,
        QtGui.QImage.Format_RGBA8888,
    )
    return image.copy()


def _array_to_qimage(
    data: ndarray, cmap: str, vmin: float | None, vmax: float | None
) -> tuple[QtGui.QImage, float, float]:
    """Convert numpy array to QImage using colormap, returning limits used."""
    norm, vmin_resolved, vmax_resolved = _normalize_data(data, vmin=vmin, vmax=vmax)
    rgba_uint8 = _apply_colormap(norm, cmap=cmap)
    height, width = norm.shape
    image = QtGui.QImage(
        rgba_uint8.data,
        width,
        height,
        QtGui.QImage.Format_RGBA8888,
    )
    return image.copy(), vmin_resolved, vmax_resolved


def _transform_mode(interpolation: str) -> QtCore.Qt.TransformationMode:
    """Map interpolation keyword to Qt transformation mode."""
    interpolation = interpolation.lower()
    if interpolation in {"nearest", "none"}:
        return QtCore.Qt.FastTransformation
    if interpolation in {"bilinear", "linear", "bicubic", "smooth"}:
        return QtCore.Qt.SmoothTransformation
    raise ValueError(
        f"Unsupported interpolation '{interpolation}' (use nearest, bilinear)"
    )


def _aspect_mode(aspect: str) -> str:
    aspect = aspect.lower()
    if aspect in {"equal", "auto"}:
        return aspect
    raise ValueError("aspect must be 'equal' or 'auto'")


def _validate_axis(axis: ndarray | None, length: int, name: str) -> ndarray | None:
    """Validate optional axis array length."""
    if axis is None:
        return None
    axis_arr = asarray(axis, dtype=float)
    if axis_arr.ndim != 1 or axis_arr.size != length:
        raise ValueError(
            f"{name} must be 1D with length matching data dimension ({length})"
        )
    return axis_arr


def _axis_span(axis: ndarray | None, length: int, name: str) -> float:
    """Return physical span for an axis or fallback to pixel length."""
    axis_arr = _validate_axis(axis, length, name)
    if axis_arr is None:
        return float(length)
    span = float(axis_arr[-1] - axis_arr[0])
    return abs(span) if span != 0 else float(length)


def _axis_limits(axis: ndarray | None, length: int) -> tuple[float, float, float]:
    """Return (min, max, span) for an axis or pixel index."""
    axis_arr = _validate_axis(axis, length, "axis")
    if axis_arr is None:
        if length <= 1:
            return 0.0, 1.0, 1.0
        return 0.0, float(length), float(length)
    start = float(axis_arr[0])
    end = float(axis_arr[-1])
    span = end - start
    if span == 0:
        span = 1.0
    return start, end, span


def _nearest_index(value: float, axis: ndarray | None, length: int) -> int:
    """Return nearest index for a value against an optional axis."""
    if length <= 0:
        return 0
    if axis is None:
        idx = int(round(value))
    else:
        idx = int(np.abs(axis - value).argmin())
    return int(clip(idx, 0, length - 1))


def _axis_value(axis: ndarray | None, idx: int) -> float:
    """Return numeric axis value for an index (or index itself)."""
    if axis is None:
        return float(idx)
    idx = int(clip(idx, 0, axis.size - 1))
    return float(axis[idx])


def _tick_values(min_val: float, max_val: float, count: int = 5) -> list[float]:
    if count <= 1:
        return [min_val]
    if min_val == max_val:
        return [min_val]
    return [min_val + (max_val - min_val) * i / (count - 1) for i in range(count)]


def _format_tick(val: float) -> str:
    """Short tick formatter similar to matplotlib defaults."""
    return f"{val:.4g}"


def _scaled_tick_labels(values: list[float]) -> tuple[list[str], str]:
    """Return labels and optional scale text using scientific notation if needed."""
    finite = [abs(v) for v in values if isfinite(v) and v != 0]
    if not finite:
        return [_format_tick(v) for v in values], ""
    max_abs = max(finite)
    exp = int(np.floor(log10(max_abs)))
    if exp >= 4 or exp <= -3:
        scale = 10**exp
        scale_text = f"×1e{exp}"
        labels = [_format_tick(v / scale) for v in values]
        return labels, scale_text
    return [_format_tick(v) for v in values], ""


def _rubber_band_colors(cmap: str) -> tuple[QtGui.QColor, QtGui.QColor]:
    """Choose rubber-band pen/fill colors contrasting with cmap mid-tone."""
    r, g, b, _ = _get_cmap(cmap)(0.5)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luminance > 0.5:
        pen_color = QtGui.QColor(20, 20, 20)
        fill_color = QtGui.QColor(20, 20, 20, 60)
    else:
        pen_color = QtGui.QColor(245, 245, 245)
        fill_color = QtGui.QColor(245, 245, 245, 60)
    return pen_color, fill_color


class _ImageCanvas(QtWidgets.QWidget):
    """Custom widget that draws image plus axes, ticks, and labels."""

    def __init__(
        self,
        data: ndarray,
        *,
        cmap: str,
        vmin: float | None,
        vmax: float | None,
        interpolation: str,
        xaxis: ndarray | None,
        yaxis: ndarray | None,
        aspect: str,
        title: str,
        xlabel: str,
        ylabel: str,
        colorbar: bool,
        colorbar_label: str,
    ) -> None:
        super().__init__()
        self._interpolation_mode = _transform_mode(interpolation)
        self._render_smooth = self._interpolation_mode == QtCore.Qt.SmoothTransformation
        self._aspect_mode = _aspect_mode(aspect)
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._colorbar = bool(colorbar) or bool(colorbar_label)
        self._colorbar_label = colorbar_label
        self._cmap = cmap
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_mode_menu)
        self._data: ndarray = asarray(data)
        self._xaxis = xaxis
        self._yaxis = yaxis
        self._qimage, self._vmin, self._vmax = _array_to_qimage(
            self._data, cmap=cmap, vmin=vmin, vmax=vmax
        )
        self._colorbar_image: QtGui.QImage = _make_colorbar_image(self._cmap)
        self._update_content_ratio()
        self._zoom_factor = 1.0
        self._view_center = QtCore.QPointF(0.5, 0.5)
        self._drag_mode: str = "pan"  # pan or box
        self._dragging = False
        self._drag_start = QtCore.QPoint()
        self._view_center_start = QtCore.QPointF(0.5, 0.5)
        self._rubber_band: QtCore.QRect | None = None
        self._markers: list[dict[str, object]] = []
        self._drag_target: dict[str, object] | None = None  # {"kind": "marker"/"box", "idx": int}
        self._box_drag_start = QtCore.QPoint()
        self._mode_button: QtWidgets.QPushButton | None = None
        self._current_layout: dict[str, object] | None = None
        self._tight_enabled = False
        self._tight_pad = 1.08
        self._tight_w_pad: float | None = None
        self._tight_h_pad: float | None = None
        self._tight_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
        self._tight_auto_resize = False
        self._tight_auto_resize_pending = False

    def _update_content_ratio(self) -> None:
        x_span = _axis_span(self._xaxis, self._data.shape[1], "xaxis")
        y_span = _axis_span(self._yaxis, self._data.shape[0], "yaxis")
        self._content_ratio = x_span / y_span if y_span != 0 else 1.0

    def _view_window(self) -> tuple[float, float, float, float]:
        """Return normalized view center and size (fractions of full image)."""
        width_frac = min(1.0, 1.0 / self._zoom_factor)
        height_frac = min(1.0, 1.0 / self._zoom_factor)
        half_w = width_frac / 2.0
        half_h = height_frac / 2.0
        cx = float(clip(self._view_center.x(), half_w, 1.0 - half_w))
        cy = float(clip(self._view_center.y(), half_h, 1.0 - half_h))
        self._view_center = QtCore.QPointF(cx, cy)
        return cx, cy, width_frac, height_frac

    def set_image(
        self,
        data: ndarray,
        *,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        xaxis: ndarray | None = None,
        yaxis: ndarray | None = None,
    ) -> None:
        """Update image data and axes."""
        arr = asarray(data)
        if arr.ndim != 2:
            raise ValueError(f"imagescqt expects a 2D array, got shape {arr.shape}")
        self._data = arr
        self._xaxis = xaxis
        self._yaxis = yaxis
        self._cmap = cmap
        self._qimage, self._vmin, self._vmax = _array_to_qimage(
            arr, cmap=cmap, vmin=vmin, vmax=vmax
        )
        self._colorbar_image = _make_colorbar_image(self._cmap)
        self._update_content_ratio()
        self._zoom_factor = 1.0
        self._view_center = QtCore.QPointF(0.5, 0.5)
        self._markers = []
        self._drag_target = None
        self._current_layout = None
        self.update()

    def set_xlabel(self, text: str) -> None:
        self._xlabel = text
        self.update()

    def set_ylabel(self, text: str) -> None:
        self._ylabel = text
        self.update()

    def set_title(self, text: str) -> None:
        self._title = text
        self.update()

    def set_colorbar(self, enabled: bool) -> None:
        self._colorbar = bool(enabled) or bool(self._colorbar_label)
        self.update()

    def set_colorbar_label(self, text: str) -> None:
        self._colorbar_label = text
        if text:
            self._colorbar = True
        self.update()

    def tight_layout(
        self,
        *,
        pad: float = 1.08,
        w_pad: float | None = None,
        h_pad: float | None = None,
        rect: tuple[float, float, float, float] | None = None,
        auto_resize: bool = True,
    ) -> None:
        """Enable matplotlib-like tight_layout that reduces unused margins."""
        self._tight_enabled = True
        self._tight_pad = float(pad)
        self._tight_w_pad = float(w_pad) if w_pad is not None else None
        self._tight_h_pad = float(h_pad) if h_pad is not None else None
        self._tight_auto_resize = bool(auto_resize)
        self._tight_auto_resize_pending = self._tight_auto_resize
        if rect is not None:
            if len(rect) != 4:
                raise ValueError("rect must be a 4-tuple (left, bottom, right, top)")
            l, b, r, t = rect
            r = max(r, l)
            t = max(t, b)
            self._tight_rect = (
                float(clip(l, 0.0, 1.0)),
                float(clip(b, 0.0, 1.0)),
                float(clip(r, 0.0, 1.0)),
                float(clip(t, 0.0, 1.0)),
            )
        else:
            self._tight_rect = (0.0, 0.0, 1.0, 1.0)
        self._current_layout = None
        self.update()

    def disable_tight_layout(self) -> None:
        """Turn off tight_layout and restore heuristic margins."""
        self._tight_enabled = False
        self._tight_auto_resize = False
        self._tight_auto_resize_pending = False
        self._current_layout = None
        self.update()

    def _layout(self, fm: QtGui.QFontMetrics):
        """Compute layout rectangles and tick labels."""
        tick_len = 6
        cbar_width = 16
        cbar_gap = 10

        x_start, x_end, x_span = _axis_limits(self._xaxis, self._data.shape[1])
        y_start, y_end, y_span = _axis_limits(self._yaxis, self._data.shape[0])
        cx, cy, width_frac, height_frac = self._view_window()

        x_view_min = x_start + (cx - width_frac / 2.0) * x_span
        x_view_max = x_start + (cx + width_frac / 2.0) * x_span
        y_view_min = y_start + (cy - height_frac / 2.0) * y_span
        y_view_max = y_start + (cy + height_frac / 2.0) * y_span

        x_ticks = _tick_values(x_view_min, x_view_max, 5)
        y_ticks = _tick_values(y_view_min, y_view_max, 5)
        x_tick_labels = [_format_tick(v) for v in x_ticks]
        y_tick_labels = [_format_tick(v) for v in y_ticks]

        cbar_ticks: list[float] = []
        cbar_tick_labels: list[str] = []
        cbar_scale_text = ""
        if self._colorbar:
            cbar_ticks = _tick_values(self._vmin, self._vmax, 5)
            cbar_tick_labels, cbar_scale_text = _scaled_tick_labels(cbar_ticks)

        max_y_label_width = max(
            (fm.horizontalAdvance(t) for t in y_tick_labels), default=0
        )
        pad_px = 6
        w_pad_px = 8
        h_pad_px = 8
        rect_left, rect_bottom, rect_right, rect_top = 0.0, 0.0, 1.0, 1.0
        if self._tight_enabled:
            pad_px = max(int(round(self._tight_pad * fm.height() * 0.5)), 2)
            w_pad_px = max(
                int(
                    round(
                        (self._tight_w_pad if self._tight_w_pad is not None else self._tight_pad)
                        * fm.height()
                        * 0.5
                    )
                ),
                2,
            )
            h_pad_px = max(
                int(
                    round(
                        (self._tight_h_pad if self._tight_h_pad is not None else self._tight_pad)
                        * fm.height()
                        * 0.5
                    )
                ),
                2,
            )
            rect_left, rect_bottom, rect_right, rect_top = self._tight_rect

        xlabel_height = fm.height() if self._xlabel else 0
        ylabel_height = fm.boundingRect(self._ylabel).height() if self._ylabel else 0
        title_height = fm.boundingRect(self._title).height() if self._title else 0
        cbar_label_height = (
            fm.boundingRect(self._colorbar_label).height() if self._colorbar_label else 0
        )

        y_tick_block = tick_len + max_y_label_width
        margin_left = y_tick_block + pad_px
        if self._ylabel:
            margin_left += w_pad_px + max(ylabel_height, fm.height())

        tick_block = tick_len + fm.height()
        margin_bottom = tick_block + max(pad_px // 2, 2)
        if self._xlabel:
            margin_bottom += h_pad_px + xlabel_height
        if cbar_scale_text:
            margin_bottom = max(
                margin_bottom, tick_block + max(pad_px // 2, 2) + 2 + fm.height()
            )

        margin_top = max(pad_px // 2, 2)
        if self._title:
            margin_top += h_pad_px + title_height
        else:
            margin_top = max(margin_top, fm.height() // 3)

        max_cbar_label_width = max(
            (fm.horizontalAdvance(t) for t in cbar_tick_labels), default=0
        )
        max_cbar_label_width = min(max_cbar_label_width, 80)
        cbar_scale_width = (
            fm.horizontalAdvance(cbar_scale_text) if cbar_scale_text else 0
        )
        cbar_scale_width = min(cbar_scale_width, 80)
        if self._colorbar:
            margin_right = (
                cbar_gap
                + cbar_width
                + tick_len
                + max(max_cbar_label_width, cbar_scale_width)
                + max(pad_px // 2, 2)
            )
            if self._colorbar_label:
                margin_right = max(
                    margin_right,
                    cbar_gap
                    + cbar_width
                    + tick_len
                    + max(max_cbar_label_width, cbar_scale_width)
                    + max(w_pad_px // 2, 2)
                    + max(cbar_label_height, fm.height()),
                )
        else:
            margin_right = pad_px + fm.height()

        if self._tight_enabled:
            fig_w = max(self.width(), 1)
            fig_h = max(self.height(), 1)
            rect_left_px = int(rect_left * fig_w)
            rect_right_px = int((1.0 - rect_right) * fig_w)
            rect_bottom_px = int(rect_bottom * fig_h)
            rect_top_px = int((1.0 - rect_top) * fig_h)
            margin_left = max(margin_left, rect_left_px)
            margin_right = max(margin_right, rect_right_px)
            margin_bottom = max(margin_bottom, rect_bottom_px)
            margin_top = max(margin_top, rect_top_px)

        margin_left = int(round(margin_left))
        margin_right = int(round(margin_right))
        margin_top = int(round(margin_top))
        margin_bottom = int(round(margin_bottom))

        avail_w = max(self.width() - margin_left - margin_right, 1)
        avail_h = max(self.height() - margin_top - margin_bottom, 1)

        if self._aspect_mode == "equal":
            desired_ratio = self._content_ratio if self._content_ratio > 0 else 1.0
            avail_ratio = avail_w / avail_h
            if avail_ratio > desired_ratio:
                draw_h = avail_h
                draw_w = int(draw_h * desired_ratio)
            else:
                draw_w = avail_w
                draw_h = int(draw_w / desired_ratio) if desired_ratio else avail_h
        else:
            draw_w = avail_w
            draw_h = avail_h

        draw_w = max(draw_w, 1)
        draw_h = max(draw_h, 1)
        draw_x = margin_left + (avail_w - draw_w) // 2
        if self._tight_enabled and self._aspect_mode == "equal":
            draw_y = margin_top
        else:
            draw_y = margin_top + (avail_h - draw_h) // 2
        draw_rect = QtCore.QRect(draw_x, draw_y, draw_w, draw_h)

        return {
            "tick_len": tick_len,
            "cbar_width": cbar_width,
            "cbar_gap": cbar_gap,
            "x_ticks": x_ticks,
            "y_ticks": y_ticks,
            "x_tick_labels": x_tick_labels,
            "y_tick_labels": y_tick_labels,
            "cbar_ticks": cbar_ticks,
            "cbar_tick_labels": cbar_tick_labels,
            "cbar_scale_text": cbar_scale_text,
            "max_cbar_label_width": max_cbar_label_width,
            "cbar_scale_width": cbar_scale_width,
            "draw_rect": draw_rect,
            "y_tick_block": y_tick_block,
            "view_limits": {
                "x_min": x_view_min,
                "x_max": x_view_max,
                "y_min": y_view_min,
                "y_max": y_view_max,
            },
            "view": {
                "cx": cx,
                "cy": cy,
                "width_frac": width_frac,
                "height_frac": height_frac,
            },
            "margins": (margin_left, margin_right, margin_top, margin_bottom),
        }

    def _pixel_from_position(
        self, pos: QtCore.QPoint, layout: dict[str, object]
    ) -> tuple[int, int] | None:
        """Map a widget position to the nearest data pixel index."""
        draw_rect: QtCore.QRect = layout["draw_rect"]
        if draw_rect.width() <= 0 or draw_rect.height() <= 0:
            return None
        if not draw_rect.contains(pos):
            return None
        view_limits: dict[str, float] = layout["view_limits"]
        x_range = view_limits["x_max"] - view_limits["x_min"]
        y_range = view_limits["y_max"] - view_limits["y_min"]
        if x_range == 0 or y_range == 0:
            return None

        rel_x = (pos.x() - draw_rect.left()) / draw_rect.width()
        rel_y = (pos.y() - draw_rect.top()) / draw_rect.height()
        x_val = view_limits["x_min"] + rel_x * x_range
        y_val = view_limits["y_min"] + rel_y * y_range
        col_idx = _nearest_index(x_val, self._xaxis, self._data.shape[1])
        row_idx = _nearest_index(y_val, self._yaxis, self._data.shape[0])
        return col_idx, row_idx

    def _add_or_move_marker(
        self,
        pos: QtCore.QPoint,
        layout: dict[str, object] | None = None,
        existing_idx: int | None = None,
    ) -> int | None:
        """Place marker at nearest pixel or move existing one."""
        fm = QtGui.QFontMetrics(self.font())
        layout = layout or self._current_layout or self._layout(fm)
        pixel = self._pixel_from_position(pos, layout)
        if pixel is None:
            return None
        if existing_idx is None:
            marker = {
                "index": pixel,
                "box_offset": None,
            }
            self._markers.append(marker)
            self._rubber_band = None
            self.update()
            return len(self._markers) - 1
        self._markers[existing_idx]["index"] = pixel
        self._rubber_band = None
        self.update()
        return existing_idx

    def _marker_position(
        self, marker: dict[str, object], layout: dict[str, object]
    ) -> tuple[QtCore.QPointF, float, float, float] | None:
        """Return screen position plus data/axis values for a marker."""
        idx = marker.get("index")
        if idx is None:
            return None
        col_idx, row_idx = idx
        if not (
            0 <= row_idx < self._data.shape[0]
            and 0 <= col_idx < self._data.shape[1]
        ):
            marker["index"] = None
            return None

        view_limits: dict[str, float] = layout["view_limits"]
        x_range = view_limits["x_max"] - view_limits["x_min"]
        y_range = view_limits["y_max"] - view_limits["y_min"]
        if x_range == 0 or y_range == 0:
            return None

        x_val = _axis_value(self._xaxis, col_idx)
        y_val = _axis_value(self._yaxis, row_idx)
        rel_x = (x_val - view_limits["x_min"]) / x_range
        rel_y = (y_val - view_limits["y_min"]) / y_range
        if rel_x < 0.0 or rel_x > 1.0 or rel_y < 0.0 or rel_y > 1.0:
            return None

        draw_rect: QtCore.QRect = layout["draw_rect"]
        px = draw_rect.left() + rel_x * draw_rect.width()
        py = draw_rect.top() + rel_y * draw_rect.height()
        data_val = float(self._data[row_idx, col_idx])
        return QtCore.QPointF(px, py), x_val, y_val, data_val

    def _marker_box_rect(
        self,
        marker: dict[str, object],
        layout: dict[str, object],
        fm: QtGui.QFontMetrics | None = None,
    ) -> tuple[QtCore.QRectF, QtCore.QPointF, list[str]] | None:
        """Compute bounding rect for marker tooltip box."""
        marker_info = self._marker_position(marker, layout)
        if marker_info is None:
            return None
        marker_pos, x_val, y_val, data_val = marker_info
        fm = fm or QtGui.QFontMetrics(self.font())
        lines = [
            f"x: {x_val:.4g}",
            f"y: {y_val:.4g}",
            f"value: {data_val:.4g}",
        ]
        text_width = max((fm.horizontalAdvance(t) for t in lines), default=0)
        text_height = fm.height() * len(lines) + 10

        box_offset = marker.get("box_offset")
        if box_offset is None:
            offset_x = 12
            offset_y = -12
            box_x = marker_pos.x() + offset_x
            box_y = marker_pos.y() + offset_y - text_height
            if box_x + text_width + 14 > self.width():
                box_x = marker_pos.x() - offset_x - text_width - 14
            if box_y < 0:
                box_y = marker_pos.y() + offset_x
            marker["box_offset"] = QtCore.QPointF(
                box_x - marker_pos.x(), box_y - marker_pos.y()
            )
        else:
            box_x = marker_pos.x() + float(box_offset.x())
            box_y = marker_pos.y() + float(box_offset.y())
            margin = 4.0
            box_x = float(
                clip(
                    box_x,
                    margin - (text_width + 14),
                    self.width() - margin,
                )
            )
            box_y = float(
                clip(
                    box_y,
                    margin,
                    self.height() - margin - text_height,
                )
            )
            marker["box_offset"] = QtCore.QPointF(
                box_x - marker_pos.x(), box_y - marker_pos.y()
            )

        box_rect = QtCore.QRectF(
            box_x,
            box_y,
            text_width + 14,
            text_height,
        )
        return box_rect, marker_pos, lines

    def _marker_hit(
        self, pos: QtCore.QPoint, layout: dict[str, object]
    ) -> tuple[int, dict[str, object]] | None:
        """Check if a click is close enough to any marker handle."""
        for idx, marker in enumerate(self._markers):
            marker_info = self._marker_position(marker, layout)
            if marker_info is None:
                continue
            marker_pos, _, _, _ = marker_info
            if hypot(pos.x() - marker_pos.x(), pos.y() - marker_pos.y()) <= 8.0:
                return idx, marker
        return None

    def _marker_box_hit(
        self, pos: QtCore.QPoint, layout: dict[str, object]
    ) -> tuple[int, dict[str, object]] | None:
        """Check if a click lands on any marker tooltip box."""
        fm = QtGui.QFontMetrics(self.font())
        for idx, marker in enumerate(self._markers):
            box_info = self._marker_box_rect(marker, layout, fm)
            if box_info is None:
                continue
            box_rect, _, _ = box_info
            if box_rect.contains(pos):
                return idx, marker
        return None

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("white"))

        fm = painter.fontMetrics()
        layout = self._layout(fm)
        self._current_layout = layout
        draw_rect = layout["draw_rect"]
        tick_len = layout["tick_len"]
        cbar_width = layout["cbar_width"]
        cbar_gap = layout["cbar_gap"]
        x_ticks = layout["x_ticks"]
        y_ticks = layout["y_ticks"]
        x_tick_labels = layout["x_tick_labels"]
        y_tick_labels = layout["y_tick_labels"]
        cbar_ticks = layout["cbar_ticks"]
        cbar_tick_labels = layout["cbar_tick_labels"]
        cbar_scale_text = layout["cbar_scale_text"]
        max_cbar_label_width = layout["max_cbar_label_width"]
        cbar_scale_width = layout["cbar_scale_width"]
        view_limits = layout["view_limits"]
        view_state = layout["view"]
        y_tick_block = layout["y_tick_block"]
        margin_left, _, margin_top, margin_bottom = layout["margins"]

        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, self._render_smooth)
        source_rect = QtCore.QRectF(
            (view_state["cx"] - view_state["width_frac"] / 2.0) * self._qimage.width(),
            (view_state["cy"] - view_state["height_frac"] / 2.0)
            * self._qimage.height(),
            self._qimage.width() * view_state["width_frac"],
            self._qimage.height() * view_state["height_frac"],
        )
        painter.drawImage(QtCore.QRectF(draw_rect), self._qimage, source_rect)

        # Axes lines
        axis_pen = QtGui.QPen(QtGui.QColor("#444"))
        painter.setPen(axis_pen)
        painter.drawLine(
            draw_rect.left(), draw_rect.top(), draw_rect.left(), draw_rect.bottom()
        )  # y axis
        painter.drawLine(
            draw_rect.left(), draw_rect.bottom(), draw_rect.right(), draw_rect.bottom()
        )  # x axis

        # X ticks
        x_range = view_limits["x_max"] - view_limits["x_min"]
        for val, label in zip(x_ticks, x_tick_labels):
            pos = (
                draw_rect.left()
                if x_range == 0
                else draw_rect.left()
                + int((val - view_limits["x_min"]) / x_range * draw_rect.width())
            )
            painter.drawLine(
                pos, draw_rect.bottom(), pos, draw_rect.bottom() + tick_len
            )
            text_rect = QtCore.QRectF(
                pos - 30,
                draw_rect.bottom() + tick_len,
                60,
                fm.height(),
            )
            painter.drawText(
                text_rect, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, label
            )

        # Y ticks
        y_range = view_limits["y_max"] - view_limits["y_min"]
        for val, label in zip(y_ticks, y_tick_labels):
            pos = (
                draw_rect.top()
                if y_range == 0
                else draw_rect.top()
                + int((val - view_limits["y_min"]) / y_range * draw_rect.height())
            )
            painter.drawLine(draw_rect.left() - tick_len, pos, draw_rect.left(), pos)
            text_rect = QtCore.QRectF(
                draw_rect.left() - tick_len - fm.horizontalAdvance(label) - 6,
                pos - fm.ascent() // 2,
                fm.horizontalAdvance(label) + 4,
                fm.height(),
            )
            painter.drawText(
                text_rect, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, label
            )

        # X label
        if self._xlabel:
            painter.drawText(
                QtCore.QRectF(
                    draw_rect.left(),
                    draw_rect.bottom() + tick_len + fm.height() + 4,
                    draw_rect.width(),
                    fm.height(),
                ),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop,
                self._xlabel,
            )

        # Y label
        if self._ylabel:
            painter.save()
            # Position ylabel in the extra left margin to avoid overlapping tick labels.
            extra_left = max(margin_left - y_tick_block, 0)
            label_center_x = draw_rect.left() - y_tick_block - extra_left / 2.0
            painter.translate(
                label_center_x,
                draw_rect.top() + draw_rect.height() / 2,
            )
            painter.rotate(-90)
            painter.drawText(
                QtCore.QRectF(
                    -draw_rect.height() / 2,
                    -fm.height() / 2,
                    draw_rect.height(),
                    fm.height(),
                ),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                self._ylabel,
            )
            painter.restore()

        # Colorbar
        if self._colorbar:
            cbar_x = draw_rect.right() + cbar_gap
            cbar_y = draw_rect.top()
            cbar_h = draw_rect.height()
            cbar_rect = QtCore.QRect(cbar_x, cbar_y, cbar_width, cbar_h)
            scaled_cbar = self._colorbar_image.scaled(
                cbar_rect.size(),
                QtCore.Qt.IgnoreAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            painter.drawImage(cbar_rect, scaled_cbar)

            for val, label in zip(cbar_ticks, cbar_tick_labels):
                pos = (
                    cbar_y
                    + int((self._vmax - val) / (self._vmax - self._vmin) * cbar_h)
                    if self._vmax != self._vmin
                    else cbar_y
                )
                painter.drawLine(
                    cbar_x + cbar_width, pos, cbar_x + cbar_width + tick_len, pos
                )
                text_rect = QtCore.QRectF(
                    cbar_x + cbar_width + tick_len,
                    pos - fm.height() / 2,
                    max_cbar_label_width + 6,
                    fm.height(),
                )
                painter.drawText(
                    text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, label
                )

            if cbar_scale_text:
                painter.drawText(
                    QtCore.QRectF(
                        cbar_x + cbar_width + tick_len,
                        cbar_y + cbar_h + 2,
                        max(max_cbar_label_width, cbar_scale_width) + 8,
                        fm.height(),
                    ),
                    QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
                    cbar_scale_text,
                )

            if self._colorbar_label:
                painter.save()
                painter.translate(
                    cbar_x + cbar_width + tick_len + max_cbar_label_width + 8,
                    cbar_y + cbar_h / 2,
                )
                painter.rotate(90)
                painter.drawText(
                    QtCore.QRectF(-cbar_h / 2, -fm.height() / 2, cbar_h, fm.height()),
                    QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                    self._colorbar_label,
                )
                painter.restore()

        # Title
        if self._title:
            painter.drawText(
                QtCore.QRectF(0, 0, self.width(), layout["margins"][2]),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom,
                self._title,
            )
        self._draw_marker(painter, layout)
        self._draw_rubber_band(painter)
        self._auto_shrink_window(layout)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        """Zoom in/out with mouse wheel."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        step = 1.1 if delta > 0 else 0.9
        self._zoom_factor = float(clip(self._zoom_factor * step, 1.0, 10.0))
        self._current_layout = None
        self._view_window()  # clamp center after zoom changes
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != QtCore.Qt.LeftButton:
            return
        fm = QtGui.QFontMetrics(self.font())
        layout = self._current_layout or self._layout(fm)

        if event.modifiers() & QtCore.Qt.ShiftModifier:
            idx = self._add_or_move_marker(event.pos(), layout)
            if idx is not None:
                self._drag_target = {"kind": "marker", "idx": idx}
                self._dragging = False
                event.accept()
                return

        hit_marker = self._marker_hit(event.pos(), layout)
        if hit_marker is not None:
            idx, _ = hit_marker
            self._drag_target = {"kind": "marker", "idx": idx}
            self._dragging = False
            event.accept()
            return

        hit_box = self._marker_box_hit(event.pos(), layout)
        if hit_box is not None:
            idx, marker = hit_box
            self._drag_target = {"kind": "box", "idx": idx}
            self._dragging = False
            self._box_drag_start = event.pos()
            box_info = self._marker_box_rect(marker, layout, fm)
            if box_info:
                box_rect, marker_pos, _ = box_info
                marker["box_offset"] = QtCore.QPointF(
                    box_rect.left() - marker_pos.x(),
                    box_rect.top() - marker_pos.y(),
                )
            event.accept()
            return

        self._drag_target = None
        self._dragging = True
        self._drag_start = event.pos()
        self._view_center_start = QtCore.QPointF(self._view_center)
        self._rubber_band = None
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._drag_target is not None:
            kind = self._drag_target.get("kind")
            idx = int(self._drag_target.get("idx", -1))
            if 0 <= idx < len(self._markers):
                marker = self._markers[idx]
                if kind == "box":
                    delta = event.pos() - self._box_drag_start
                    current_offset: QtCore.QPointF = marker.get("box_offset") or QtCore.QPointF(
                        0, 0
                    )
                    marker["box_offset"] = QtCore.QPointF(
                        current_offset.x() + delta.x(), current_offset.y() + delta.y()
                    )
                    self._box_drag_start = event.pos()
                    self.update()
                    event.accept()
                    return
                if kind == "marker":
                    self._add_or_move_marker(event.pos(), existing_idx=idx)
                    event.accept()
                    return
        if not self._dragging:
            return
        if self._drag_mode == "pan":
            delta = event.pos() - self._drag_start
            layout = self._current_layout or self._layout(
                QtGui.QFontMetrics(self.font())
            )
            draw_rect: QtCore.QRect = layout["draw_rect"]
            if draw_rect.width() > 0 and draw_rect.height() > 0:
                width_frac = min(1.0, 1.0 / self._zoom_factor)
                height_frac = min(1.0, 1.0 / self._zoom_factor)
                dx = delta.x() / draw_rect.width() * width_frac
                dy = delta.y() / draw_rect.height() * height_frac
                self._view_center = QtCore.QPointF(
                    self._view_center_start.x() - dx,
                    self._view_center_start.y() - dy,
                )
                self._view_window()
                self._current_layout = None
            self.update()
        else:  # box
            rect = QtCore.QRect(self._drag_start, event.pos()).normalized()
            self._rubber_band = rect
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self._drag_target is not None:
            self._drag_target = None
            event.accept()
            return
        if not self._dragging:
            return
        self._dragging = False
        if self._drag_mode == "box" and self._rubber_band is not None:
            self._apply_box_zoom(self._rubber_band)
            self._rubber_band = None
        self._current_layout = None
        self.update()
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == QtCore.Qt.Key_S and (
            event.modifiers() & QtCore.Qt.ShiftModifier
        ):
            self._drag_mode = "box" if self._drag_mode == "pan" else "pan"
            event.accept()
            return
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            if self._markers:
                self._clear_marker(target_idx=None)
                event.accept()
                return
        super().keyPressEvent(event)

    def _show_mode_menu(self, pos: QtCore.QPoint) -> None:
        """Context menu to toggle drag mode or clear marker."""
        fm = QtGui.QFontMetrics(self.font())
        layout = self._current_layout or self._layout(fm)
        menu = QtWidgets.QMenu(self)
        pan_action = menu.addAction("Pan mode")
        box_action = menu.addAction("Box-zoom mode")
        delete_action = None
        hit_marker = self._marker_hit(pos, layout)
        hit_box = self._marker_box_hit(pos, layout)
        target_idx = None
        if hit_marker is not None:
            target_idx = hit_marker[0]
        elif hit_box is not None:
            target_idx = hit_box[0]
        if target_idx is not None:
            delete_action = menu.addAction("Delete marker")
        current_mode = self._drag_mode
        pan_action.setCheckable(True)
        box_action.setCheckable(True)
        if current_mode == "pan":
            pan_action.setChecked(True)
        else:
            box_action.setChecked(True)

        chosen = menu.exec(self.mapToGlobal(pos))
        if chosen == pan_action:
            self._drag_mode = "pan"
        elif chosen == box_action:
            self._drag_mode = "box"
        elif delete_action is not None and chosen == delete_action and target_idx is not None:
            self._clear_marker(target_idx)

    def _apply_box_zoom(self, selection: QtCore.QRect) -> None:
        """Zoom into the selected rectangle."""
        fm = QtGui.QFontMetrics(self.font())
        layout = self._current_layout or self._layout(fm)
        draw_rect: QtCore.QRect = layout["draw_rect"]
        intersect = selection & draw_rect
        if intersect.isEmpty():
            return

        sel_w = max(intersect.width(), 1)
        sel_h = max(intersect.height(), 1)
        draw_w = max(draw_rect.width(), 1)
        draw_h = max(draw_rect.height(), 1)

        center_norm_x = (intersect.center().x() - draw_rect.left()) / draw_w
        center_norm_y = (intersect.center().y() - draw_rect.top()) / draw_h
        view_state = layout["view"]
        view_left = view_state["cx"] - view_state["width_frac"] / 2.0
        view_top = view_state["cy"] - view_state["height_frac"] / 2.0
        new_center = QtCore.QPointF(
            view_left + center_norm_x * view_state["width_frac"],
            view_top + center_norm_y * view_state["height_frac"],
        )

        factor_w = draw_w / sel_w
        factor_h = draw_h / sel_h
        new_zoom = float(clip(self._zoom_factor * min(factor_w, factor_h), 1.0, 20.0))

        self._view_center = new_center
        self._zoom_factor = new_zoom
        self._view_window()
        self._current_layout = None

    def _clear_marker(self, target_idx: int | None = None) -> None:
        """Remove marker(s) and tooltips."""
        if target_idx is None:
            self._markers.clear()
        else:
            if 0 <= target_idx < len(self._markers):
                self._markers.pop(target_idx)
        self._drag_target = None
        self.update()

    def _draw_marker(self, painter: QtGui.QPainter, layout: dict[str, object]) -> None:
        """Render MATLAB-style data markers and tooltips."""
        if not self._markers:
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        fm = painter.fontMetrics()
        highlight = QtGui.QColor("#d32f2f")

        for marker in self._markers:
            box_info = self._marker_box_rect(marker, layout, fm)
            if box_info is None:
                continue
            box_rect, marker_pos, lines = box_info

            outer_pen = QtGui.QPen(QtGui.QColor("white"))
            outer_pen.setWidth(4)
            painter.setPen(outer_pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(marker_pos, 6, 6)

            inner_pen = QtGui.QPen(highlight)
            inner_pen.setWidth(2)
            painter.setPen(inner_pen)
            painter.drawEllipse(marker_pos, 5, 5)
            painter.drawLine(
                marker_pos + QtCore.QPointF(-10, 0),
                marker_pos + QtCore.QPointF(10, 0),
            )
            painter.drawLine(
                marker_pos + QtCore.QPointF(0, -10),
                marker_pos + QtCore.QPointF(0, 10),
            )

            painter.setPen(QtGui.QPen(highlight, 1.5))
            anchor_point = QtCore.QPointF(
                box_rect.left() + 8, box_rect.top() + box_rect.height() / 2
            )
            painter.drawLine(marker_pos, anchor_point)

            painter.setBrush(QtGui.QColor(255, 255, 255, 235))
            painter.setPen(QtGui.QPen(QtGui.QColor("#444"), 1))
            painter.drawRoundedRect(box_rect, 6, 6)

            text_y = box_rect.top() + 6 + fm.ascent()
            painter.setPen(QtGui.QPen(QtGui.QColor("#111")))
            for line in lines:
                painter.drawText(box_rect.left() + 6, text_y, line)
                text_y += fm.height()

        painter.restore()

    def _draw_rubber_band(self, painter: QtGui.QPainter) -> None:
        if self._drag_mode != "box" or self._rubber_band is None:
            return
        pen_color, fill_color = _rubber_band_colors(self._cmap)
        pen = QtGui.QPen(pen_color)
        pen.setStyle(QtCore.Qt.DashLine)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(fill_color))
        painter.drawRect(self._rubber_band)

    def _auto_shrink_window(self, layout: dict[str, object]) -> None:
        """Optionally shrink the parent window to fit content more tightly."""
        if not (self._tight_enabled and self._tight_auto_resize_pending):
            return
        win = self.window()
        if win is None:
            return

        margins = layout["margins"]
        draw_rect: QtCore.QRect = layout["draw_rect"]
        avail_w = self.width() - margins[0] - margins[1]
        avail_h = self.height() - margins[2] - margins[3]
        extra_w = max(avail_w - draw_rect.width(), 0)
        extra_h = max(avail_h - draw_rect.height(), 0)
        tol = 8
        shrink_w = extra_w - tol if extra_w > tol else 0
        shrink_h = extra_h - tol if extra_h > tol else 0
        if shrink_w <= 0 and shrink_h <= 0:
            self._tight_auto_resize_pending = False
            return

        frame_dx = win.width() - self.width()
        frame_dy = win.height() - self.height()
        new_w = max(win.width() - shrink_w, self.minimumWidth() + frame_dx)
        new_h = max(win.height() - shrink_h, self.minimumHeight() + frame_dy)
        if new_w < win.width() or new_h < win.height():
            win.resize(int(new_w), int(new_h))
        self._tight_auto_resize_pending = False


class ImageWindow(QtWidgets.QMainWindow):
    """Basic window that displays a single 2D numpy array as an image."""

    def __init__(
        self,
        data: ndarray,
        *,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        title: str = "imagescqt",
        interpolation: str = "nearest",
        xaxis: ndarray | None = None,
        yaxis: ndarray | None = None,
        aspect: str = "equal",
        xlabel: str = "",
        ylabel: str = "",
        colorbar: bool = False,
        colorbar_label: str = "",
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self._canvas = _ImageCanvas(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
            xaxis=xaxis,
            yaxis=yaxis,
            aspect=aspect,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            colorbar=colorbar,
            colorbar_label=colorbar_label,
        )
        self.setCentralWidget(self._canvas)
        self.resize(720, 540)

    def set_image(
        self,
        data: ndarray,
        *,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        xaxis: ndarray | None = None,
        yaxis: ndarray | None = None,
    ) -> None:
        """Update the displayed image and optional axes."""
        self._canvas.set_image(
            data, cmap=cmap, vmin=vmin, vmax=vmax, xaxis=xaxis, yaxis=yaxis
        )

    def set_xlabel(self, text: str) -> None:
        """Set x-axis label."""
        self._canvas.set_xlabel(text)

    def set_ylabel(self, text: str) -> None:
        """Set y-axis label."""
        self._canvas.set_ylabel(text)

    def set_title(self, text: str) -> None:
        """Set figure title (also updates window title)."""
        self._canvas.set_title(text)
        self.setWindowTitle(text)

    def add_colorbar(self, label: str = "") -> None:
        """Enable colorbar and optionally set its label."""
        self._canvas.set_colorbar(True)
        if label:
            self._canvas.set_colorbar_label(label)

    def set_colorbar(self, enabled: bool) -> None:
        """Show or hide colorbar."""
        self._canvas.set_colorbar(enabled)

    def set_colorbar_label(self, text: str) -> None:
        """Set colorbar label (enables colorbar if provided)."""
        self._canvas.set_colorbar_label(text)

    def tight_layout(
        self,
        *,
        pad: float = 1.08,
        w_pad: float | None = None,
        h_pad: float | None = None,
        rect: tuple[float, float, float, float] | None = None,
        auto_resize: bool = True,
    ) -> None:
        """Enable matplotlib-like tight_layout on the canvas."""
        self._canvas.tight_layout(
            pad=pad, w_pad=w_pad, h_pad=h_pad, rect=rect, auto_resize=auto_resize
        )

    def disable_tight_layout(self) -> None:
        """Disable tight_layout and revert to default spacing."""
        self._canvas.disable_tight_layout()


def imagescqt(
    data: ndarray,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "imagescqt",
    interpolation: str = "nearest",
    xaxis: ndarray | None = None,
    yaxis: ndarray | None = None,
    aspect: str = "equal",
    xlabel: str = "",
    ylabel: str = "",
    colorbar: bool = False,
    colorbar_label: str = "",
) -> ImageWindow:
    """
    Display a numpy array in a PySide6 window (MATLAB imagesc style).

    When an application instance already exists (e.g. IPython with "%gui qt"),
    the window is shown and control returns immediately. If no application is
    running, a new one is created and the call will block until the window
    closes.
    """
    arr = asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"imagescqt expects a 2D array, got shape {arr.shape}")

    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        app = QtWidgets.QApplication([])
        created_app = True

    window = ImageWindow(
        arr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title=title,
        interpolation=interpolation,
        xaxis=xaxis,
        yaxis=yaxis,
        aspect=aspect,
        xlabel=xlabel,
        ylabel=ylabel,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
    )
    window.show()

    if created_app:
        app.exec()
    return window
