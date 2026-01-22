"""Lightweight PySide6 viewer similar to MATLAB's imagesc with a Qt toolbar."""

import warnings
from dataclasses import dataclass
from math import cos, hypot, pi, sin
from typing import Protocol

import numpy as np
from matplotlib import cm
from numpy import asarray, clip, isfinite, linspace, log10, nan_to_num, ndarray, uint8

_COLORMAP_CACHE: dict[str, object] = {}
_TICK_LEN = 6
_COLORBAR_WIDTH = 16
_COLORBAR_GAP = 10
_MARKER_OUTER_WIDTH = 4
_MARKER_INNER_WIDTH = 2
_MARKER_RADIUS_OUTER = 6
_MARKER_RADIUS_INNER = 5
_MARKER_CROSS_HALF = 10
_MARKER_HIT_RADIUS = 8.0
_MARKER_BOX_OFFSET = 12
_MARKER_TOOLTIP_H_PADDING = 7
_MARKER_TOOLTIP_V_PADDING = 5
_MARKER_TOOLTIP_CLAMP_MARGIN = 4.0
_MARKER_ANCHOR_OFFSET = 8
_MARKER_TOOLTIP_CORNER_RADIUS = 6
_TOOLBAR_ICON_SIZE = 20
_TOOLBAR_ICON_STROKE = 1.6
_THEME_PROP = "qtplotlibTheme"
_THEME_DARK = "dark"
_THEME_LIGHT = "light"

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:
    raise ImportError("PySide6 is required to use imagescqt.") from exc


def _color_luminance(color: QtGui.QColor) -> float:
    r, g, b, _ = color.getRgbF()
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _normalize_theme(theme: str | None) -> str:
    if not theme:
        return _THEME_DARK
    value = theme.lower()
    if value not in {_THEME_DARK, _THEME_LIGHT}:
        return _THEME_DARK
    return value


def _theme_palette(theme: str) -> QtGui.QPalette:
    resolved = _normalize_theme(theme)
    palette = QtGui.QPalette()

    if resolved == _THEME_DARK:
        window = QtGui.QColor(32, 33, 36)
        base = QtGui.QColor(24, 24, 24)
        alt_base = QtGui.QColor(38, 38, 38)
        text = QtGui.QColor(235, 235, 235)
        button = QtGui.QColor(45, 45, 48)
        highlight = QtGui.QColor(61, 136, 220)
        highlight_text = QtGui.QColor(255, 255, 255)
        tooltip_base = QtGui.QColor(45, 45, 48)
        tooltip_text = QtGui.QColor(245, 245, 245)
        link = QtGui.QColor(94, 175, 255)
        placeholder = QtGui.QColor(160, 160, 160)
        disabled = QtGui.QColor(120, 120, 120)
    else:
        window = QtGui.QColor(245, 246, 248)
        base = QtGui.QColor(255, 255, 255)
        alt_base = QtGui.QColor(235, 236, 238)
        text = QtGui.QColor(28, 28, 28)
        button = QtGui.QColor(240, 240, 240)
        highlight = QtGui.QColor(49, 114, 201)
        highlight_text = QtGui.QColor(255, 255, 255)
        tooltip_base = QtGui.QColor(255, 255, 255)
        tooltip_text = QtGui.QColor(32, 32, 32)
        link = QtGui.QColor(34, 105, 200)
        placeholder = QtGui.QColor(120, 120, 120)
        disabled = QtGui.QColor(150, 150, 150)

    palette.setColor(QtGui.QPalette.Window, window)
    palette.setColor(QtGui.QPalette.WindowText, text)
    palette.setColor(QtGui.QPalette.Base, base)
    palette.setColor(QtGui.QPalette.AlternateBase, alt_base)
    palette.setColor(QtGui.QPalette.Text, text)
    palette.setColor(QtGui.QPalette.Button, button)
    palette.setColor(QtGui.QPalette.ButtonText, text)
    palette.setColor(QtGui.QPalette.ToolTipBase, tooltip_base)
    palette.setColor(QtGui.QPalette.ToolTipText, tooltip_text)
    palette.setColor(QtGui.QPalette.Highlight, highlight)
    palette.setColor(QtGui.QPalette.HighlightedText, highlight_text)
    palette.setColor(QtGui.QPalette.Link, link)
    palette.setColor(QtGui.QPalette.LinkVisited, link)
    palette.setColor(QtGui.QPalette.PlaceholderText, placeholder)
    palette.setColor(
        QtGui.QPalette.Disabled, QtGui.QPalette.Text, disabled
    )
    palette.setColor(
        QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, disabled
    )
    palette.setColor(
        QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, disabled
    )
    return palette


def _apply_theme_to_widget(root: QtWidgets.QWidget, theme: str) -> str:
    resolved = _normalize_theme(theme)
    palette = _theme_palette(resolved)
    root.setPalette(palette)
    root.setProperty(_THEME_PROP, resolved)
    root.setAutoFillBackground(True)

    if isinstance(root, QtWidgets.QMainWindow):
        central = root.centralWidget()
        if central is not None:
            central.setPalette(palette)
            central.setAutoFillBackground(True)
            central.update()

    for toolbar in root.findChildren(QtWidgets.QToolBar):
        toolbar.setPalette(palette)
        toolbar.setAutoFillBackground(True)
        if hasattr(toolbar, "set_theme_state"):
            toolbar.set_theme_state(resolved)
        if hasattr(toolbar, "refresh_icons"):
            toolbar.refresh_icons()
        toolbar.update()

    root.update()
    return resolved


def _current_theme(widget: QtWidgets.QWidget | None) -> str:
    if widget is None:
        return _THEME_DARK
    theme = widget.property(_THEME_PROP)
    if isinstance(theme, str):
        return _normalize_theme(theme)
    window = widget.window()
    if window is not None and window is not widget:
        theme = window.property(_THEME_PROP)
        if isinstance(theme, str):
            return _normalize_theme(theme)
    return _THEME_DARK


def _prepare_image_data(data: ndarray) -> ndarray:
    """Validate and coerce incoming image data for display."""
    arr = asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"imagescqt expects a 2D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("imagescqt requires a non-empty 2D array")
    if np.iscomplexobj(arr):
        warnings.warn(
            "Complex data provided to imagescqt; using magnitude for display.",
            RuntimeWarning,
            stacklevel=2,
        )
        arr = np.abs(arr)
    return arr.astype(float, copy=False)


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
    prepared = _prepare_image_data(data)
    norm, vmin_resolved, vmax_resolved = _normalize_data(
        prepared, vmin=vmin, vmax=vmax
    )
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


def _parse_axes_and_data(
    args: tuple[object, ...],
    xaxis_kw: ndarray | None,
    yaxis_kw: ndarray | None,
    func_name: str = "imagescqt",
) -> tuple[object, ndarray | None, ndarray | None]:
    """Support MATLAB-style signatures: (data) or (x, y, data)."""
    if len(args) == 1:
        return args[0], xaxis_kw, yaxis_kw
    if len(args) == 3:
        if xaxis_kw is not None or yaxis_kw is not None:
            raise TypeError(
                f"{func_name} accepts axes either positionally (x, y, data) "
                "or via xaxis/yaxis keywords, not both."
            )
        xaxis_arg, yaxis_arg, data_arg = args
        return data_arg, xaxis_arg, yaxis_arg
    if len(args) == 0:
        raise TypeError(f"{func_name} requires at least a data array.")
    raise TypeError(
        f"{func_name} supports {func_name}(data) or {func_name}(x, y, data)."
    )


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


@dataclass(frozen=True)
class _ViewLimits:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class _View:
    cx: float
    cy: float
    width_frac: float
    height_frac: float


@dataclass(frozen=True)
class _Layout:
    tick_len: int
    cbar_width: int
    cbar_gap: int
    x_ticks: list[float]
    y_ticks: list[float]
    x_tick_labels: list[str]
    y_tick_labels: list[str]
    cbar_ticks: list[float]
    cbar_tick_labels: list[str]
    cbar_scale_text: str
    max_cbar_label_width: int
    cbar_scale_width: int
    draw_rect: QtCore.QRect
    y_tick_block: int
    view_limits: _ViewLimits
    view: _View
    margins: tuple[int, int, int, int]


@dataclass
class _Marker:
    """Simple carrier for marker position and tooltip offset."""

    index: tuple[int, int] | None
    box_offset: QtCore.QPointF | None = None


@dataclass(frozen=True)
class _ToolbarActionSpec:
    key: str
    text: str
    tooltip: str
    icon_name: str
    shortcut: str | None = None
    checkable: bool = False
    separator_before: bool = False


class _ToolbarController(Protocol):
    def reset_view(self) -> None:
        """Reset the view and interaction modes to defaults."""

    def set_zoom_axis_mode(self, mode: str) -> None:
        """Switch zoom mode (off, xy, x, y)."""

    @property
    def zoom_axis_mode(self) -> str:
        """Return the current zoom mode."""

    def copy_axes_to_clipboard(self) -> None:
        """Copy only the axes region to the clipboard."""

    def copy_figure_to_clipboard(self) -> None:
        """Copy the full figure area to the clipboard."""

    def save_figure_to_png(self) -> None:
        """Save the full figure area as a PNG image."""


def _pt(size: int, x: float, y: float) -> QtCore.QPointF:
    return QtCore.QPointF(x * size, y * size)


def _rect(size: int, x: float, y: float, w: float, h: float) -> QtCore.QRectF:
    return QtCore.QRectF(x * size, y * size, w * size, h * size)


def _draw_home_icon(painter: QtGui.QPainter, size: int) -> None:
    roof = QtGui.QPainterPath()
    roof.moveTo(_pt(size, 0.2, 0.56))
    roof.lineTo(_pt(size, 0.5, 0.24))
    roof.lineTo(_pt(size, 0.8, 0.56))
    painter.drawPath(roof)
    painter.drawRect(_rect(size, 0.3, 0.56, 0.4, 0.3))
    painter.drawRect(_rect(size, 0.47, 0.67, 0.12, 0.19))


def _draw_zoom_icon(painter: QtGui.QPainter, size: int, axis: str) -> None:
    center = _pt(size, 0.42, 0.42)
    radius = 0.22 * size
    painter.drawEllipse(center, radius, radius)
    painter.drawLine(_pt(size, 0.58, 0.58), _pt(size, 0.82, 0.82))
    if axis in {"xy", "both"}:
        painter.drawLine(_pt(size, 0.32, 0.42), _pt(size, 0.52, 0.42))
        painter.drawLine(_pt(size, 0.42, 0.32), _pt(size, 0.42, 0.52))
    elif axis == "x":
        painter.drawLine(_pt(size, 0.32, 0.42), _pt(size, 0.52, 0.42))
    elif axis == "y":
        painter.drawLine(_pt(size, 0.42, 0.32), _pt(size, 0.42, 0.52))


def _draw_clipboard_icon(painter: QtGui.QPainter, size: int, *, detail: bool) -> None:
    painter.drawRoundedRect(_rect(size, 0.28, 0.3, 0.44, 0.52), 2.2, 2.2)
    painter.drawRoundedRect(_rect(size, 0.38, 0.18, 0.24, 0.14), 1.6, 1.6)
    if detail:
        painter.drawRect(_rect(size, 0.34, 0.44, 0.32, 0.26))


def _draw_save_icon(painter: QtGui.QPainter, size: int) -> None:
    painter.drawRoundedRect(_rect(size, 0.2, 0.22, 0.6, 0.56), 2.2, 2.2)
    painter.drawRect(_rect(size, 0.3, 0.26, 0.4, 0.16))
    painter.drawEllipse(_rect(size, 0.37, 0.52, 0.18, 0.18))


def _draw_sun_icon(painter: QtGui.QPainter, size: int) -> None:
    center = QtCore.QPointF(0.5 * size, 0.5 * size)
    radius = 0.18 * size
    painter.drawEllipse(center, radius, radius)
    inner = radius + 0.06 * size
    outer = radius + 0.2 * size
    for idx in range(8):
        angle = idx * (pi / 4.0)
        start = QtCore.QPointF(
            center.x() + cos(angle) * inner, center.y() + sin(angle) * inner
        )
        end = QtCore.QPointF(
            center.x() + cos(angle) * outer, center.y() + sin(angle) * outer
        )
        painter.drawLine(start, end)


def _draw_moon_icon(painter: QtGui.QPainter, size: int) -> None:
    outer = _rect(size, 0.2, 0.18, 0.6, 0.6)
    inner = _rect(size, 0.34, 0.18, 0.6, 0.6)
    start_angle = 40 * 16
    span_angle = 280 * 16
    painter.drawArc(outer, start_angle, span_angle)
    painter.drawArc(inner, start_angle, span_angle)


def _make_toolbar_icon(name: str, size: int, color: QtGui.QColor) -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    stroke = max(_TOOLBAR_ICON_STROKE, size * 0.08)
    pen = QtGui.QPen(color, stroke, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.NoBrush)
    if name == "home":
        _draw_home_icon(painter, size)
    elif name == "zoom":
        _draw_zoom_icon(painter, size, "xy")
    elif name == "zoom_x":
        _draw_zoom_icon(painter, size, "x")
    elif name == "zoom_y":
        _draw_zoom_icon(painter, size, "y")
    elif name == "copy_axes":
        _draw_clipboard_icon(painter, size, detail=True)
    elif name == "copy_figure":
        _draw_clipboard_icon(painter, size, detail=False)
    elif name == "save_png":
        _draw_save_icon(painter, size)
    elif name == "theme_light":
        _draw_sun_icon(painter, size)
    elif name == "theme_dark":
        _draw_moon_icon(painter, size)
    else:
        painter.drawRect(_rect(size, 0.2, 0.2, 0.6, 0.6))
    painter.end()
    return QtGui.QIcon(pixmap)


class FigureToolbar(QtWidgets.QToolBar):
    """Matplotlib-like toolbar with extensible action specs."""

    def __init__(
        self,
        controller: _ToolbarController,
        *,
        icon_size: int = _TOOLBAR_ICON_SIZE,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__("Figure Toolbar", parent)
        self._controller = controller
        self._zoom_actions: dict[str, QtGui.QAction] = {}
        self._actions: dict[str, QtGui.QAction] = {}
        self._icon_names: dict[str, str] = {}
        self._syncing = False

        self.setIconSize(QtCore.QSize(icon_size, icon_size))
        self.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.setMovable(False)
        self._build_actions(icon_size)
        self._sync_zoom_actions(getattr(controller, "zoom_axis_mode", "off"))
        self.set_theme_state(_current_theme(self.window()))

        zoom_signal = getattr(controller, "zoomModeChanged", None)
        if zoom_signal is not None and hasattr(zoom_signal, "connect"):
            zoom_signal.connect(self._sync_zoom_actions)

    def _build_actions(self, icon_size: int) -> None:
        palette = self.palette()
        base_color = palette.color(QtGui.QPalette.Text)
        specs = [
            _ToolbarActionSpec(
                key="home",
                text="Home",
                tooltip="Reset view",
                icon_name="home",
                shortcut="R",
            ),
            _ToolbarActionSpec(
                key="zoom",
                text="Zoom",
                tooltip="Zoom (both axes)",
                icon_name="zoom",
                checkable=True,
                shortcut="Z",
            ),
            _ToolbarActionSpec(
                key="zoom_x",
                text="Zoom X",
                tooltip="Zoom horizontally",
                icon_name="zoom_x",
                checkable=True,
                shortcut="H",
            ),
            _ToolbarActionSpec(
                key="zoom_y",
                text="Zoom Y",
                tooltip="Zoom vertically",
                icon_name="zoom_y",
                checkable=True,
                shortcut="V",
            ),
            _ToolbarActionSpec(
                key="copy_axes",
                text="Copy Axes",
                tooltip="Copy axes area to clipboard",
                icon_name="copy_axes",
                separator_before=True,
                shortcut="Shift+Y",
            ),
            _ToolbarActionSpec(
                key="copy_figure",
                text="Copy Figure",
                tooltip="Copy full figure to clipboard",
                icon_name="copy_figure",
                shortcut="Y",
            ),
            _ToolbarActionSpec(
                key="save_png",
                text="Save PNG",
                tooltip="Save figure as PNG",
                icon_name="save_png",
                shortcut="S",
            ),
            _ToolbarActionSpec(
                key="theme",
                text="Theme",
                tooltip="Switch to light theme",
                icon_name="theme_light",
                separator_before=True,
                shortcut="T",
            ),
        ]

        for spec in specs:
            if spec.separator_before:
                self.addSeparator()
            icon = _make_toolbar_icon(spec.icon_name, icon_size, base_color)
            action = QtGui.QAction(icon, spec.text, self)
            action.setCheckable(spec.checkable)
            if spec.shortcut:
                sequence = QtGui.QKeySequence(spec.shortcut)
                action.setShortcut(sequence)
                action.setShortcutContext(QtCore.Qt.WindowShortcut)
            self._apply_action_tooltip(action, spec.tooltip)
            self.addAction(action)
            self._actions[spec.key] = action
            self._icon_names[spec.key] = spec.icon_name
            if spec.key in {"zoom", "zoom_x", "zoom_y"}:
                self._zoom_actions[spec.key] = action

        self._actions["home"].triggered.connect(self._handle_home)
        self._actions["zoom"].toggled.connect(lambda checked: self._handle_zoom("xy", checked))
        self._actions["zoom_x"].toggled.connect(lambda checked: self._handle_zoom("x", checked))
        self._actions["zoom_y"].toggled.connect(lambda checked: self._handle_zoom("y", checked))
        self._actions["copy_axes"].triggered.connect(self._controller.copy_axes_to_clipboard)
        self._actions["copy_figure"].triggered.connect(self._controller.copy_figure_to_clipboard)
        self._actions["save_png"].triggered.connect(self._controller.save_figure_to_png)
        self._actions["theme"].triggered.connect(self._handle_theme_toggle)

    def _apply_action_tooltip(self, action: QtGui.QAction, base: str) -> None:
        sequence = action.shortcut()
        if not sequence.isEmpty():
            tooltip = f"{base} ({sequence.toString(QtGui.QKeySequence.NativeText)})"
        else:
            tooltip = base
        action.setToolTip(tooltip)
        action.setStatusTip(tooltip)

    def _update_action_icon(self, key: str) -> None:
        action = self._actions.get(key)
        icon_name = self._icon_names.get(key)
        if action is None or icon_name is None:
            return
        size = self.iconSize().width()
        base_color = self.palette().color(QtGui.QPalette.Text)
        action.setIcon(_make_toolbar_icon(icon_name, size, base_color))

    def refresh_icons(self) -> None:
        for key in self._actions:
            self._update_action_icon(key)

    def set_theme_state(self, theme: str) -> None:
        resolved = _normalize_theme(theme)
        if resolved == _THEME_DARK:
            icon_name = "theme_light"
            tooltip = "Switch to light theme"
        else:
            icon_name = "theme_dark"
            tooltip = "Switch to dark theme"
        self._icon_names["theme"] = icon_name
        action = self._actions.get("theme")
        if action is None:
            return
        self._apply_action_tooltip(action, tooltip)
        self._update_action_icon("theme")

    def _handle_theme_toggle(self) -> None:
        window = self.window()
        if window is None:
            return
        toggle = getattr(window, "toggle_theme", None)
        if callable(toggle):
            toggle()
            return
        current = _current_theme(window)
        next_theme = _THEME_LIGHT if current == _THEME_DARK else _THEME_DARK
        _apply_theme_to_widget(window, next_theme)

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        if event.type() == QtCore.QEvent.PaletteChange:
            self.refresh_icons()
        super().changeEvent(event)

    def _handle_home(self) -> None:
        self._controller.reset_view()
        self._sync_zoom_actions(getattr(self._controller, "zoom_axis_mode", "off"))

    def _handle_zoom(self, mode: str, checked: bool) -> None:
        if self._syncing:
            return
        if checked:
            self._controller.set_zoom_axis_mode(mode)
            self._sync_zoom_actions(mode)
        else:
            current = getattr(self._controller, "zoom_axis_mode", "off")
            if current == mode:
                self._controller.set_zoom_axis_mode("off")
            self._sync_zoom_actions(getattr(self._controller, "zoom_axis_mode", "off"))

    def _sync_zoom_actions(self, mode: str) -> None:
        if self._syncing:
            return
        self._syncing = True
        if mode not in {"xy", "x", "y"}:
            mode = "off"
        mapping = {"xy": "zoom", "x": "zoom_x", "y": "zoom_y"}
        for key, action in self._zoom_actions.items():
            action.setChecked(mapping.get(mode) == key)
        self._syncing = False


class _ImageCanvas(QtWidgets.QWidget):
    """Custom widget that draws image plus axes, ticks, and labels."""

    zoomModeChanged = QtCore.Signal(str)

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
        self._data: ndarray = _prepare_image_data(data)
        self._xaxis = xaxis
        self._yaxis = yaxis
        self._qimage, self._vmin, self._vmax = _array_to_qimage(
            self._data, cmap=cmap, vmin=vmin, vmax=vmax
        )
        self._colorbar_image: QtGui.QImage = _make_colorbar_image(self._cmap)
        self._update_content_ratio()
        self._reset_zoom_state()
        self._drag_mode: str = "pan"  # pan or box
        self._zoom_axis_mode: str = "off"  # off, xy, x, y
        self._dragging = False
        self._drag_start = QtCore.QPoint()
        self._view_center_start = QtCore.QPointF(0.5, 0.5)
        self._rubber_band: QtCore.QRect | None = None
        self._rubber_band_kind: str | None = None  # box or zoom-{axis}
        self._markers: list[_Marker] = []
        self._drag_target: dict[str, object] | None = None  # {"kind": "marker"/"box", "idx": int}
        self._box_drag_start = QtCore.QPoint()
        self._zoom_alt: bool = False
        self._current_layout: _Layout | None = None
        self._tight_enabled = False
        self._tight_pad = 1.08
        self._tight_w_pad: float | None = None
        self._tight_h_pad: float | None = None
        self._tight_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
        self._tight_auto_resize = False
        self._tight_auto_resize_pending = False
        self._clipboard_shortcuts: list[QtGui.QShortcut] = []
        self._toast_label: QtWidgets.QLabel | None = None
        self._toast_opacity: QtWidgets.QGraphicsOpacityEffect | None = None
        self._toast_animation: QtCore.QAbstractAnimation | None = None
        self._auto_resize_timer: QtCore.QTimer | None = None
        self._init_clipboard_shortcuts()

    def _update_content_ratio(self) -> None:
        x_span = _axis_span(self._xaxis, self._data.shape[1], "xaxis")
        y_span = _axis_span(self._yaxis, self._data.shape[0], "yaxis")
        self._content_ratio = x_span / y_span if y_span != 0 else 1.0

    def _reset_zoom_state(self) -> None:
        """Return zoom and center to the full-view defaults."""
        self._zoom_x = 1.0
        self._zoom_y = 1.0
        self._view_center = QtCore.QPointF(0.5, 0.5)

    def _update_zoom_cursor(self) -> None:
        """Reflect the current zoom axis mode in the mouse cursor."""
        if self._zoom_axis_mode == "xy":
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        elif self._zoom_axis_mode == "x":
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.SizeHorCursor))
        elif self._zoom_axis_mode == "y":
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.SizeVerCursor))
        else:
            self.unsetCursor()

    def _set_zoom_mode(self, mode: str) -> None:
        """Switch zoom axis mode; accepted values: off, xy, x, y."""
        if mode not in {"off", "xy", "x", "y"}:
            mode = "off"
        self._zoom_axis_mode = mode
        self._dragging = False
        self._rubber_band = None
        self._rubber_band_kind = None
        self._zoom_alt = False
        self._update_zoom_cursor()
        self.zoomModeChanged.emit(self._zoom_axis_mode)

    @property
    def zoom_axis_mode(self) -> str:
        """Return the current zoom mode (off, xy, x, y)."""
        return self._zoom_axis_mode

    def set_zoom_axis_mode(self, mode: str) -> None:
        """Set the active zoom axis mode."""
        self._set_zoom_mode(mode)

    def reset_view(self) -> None:
        """Reset zoom and interaction settings to their defaults."""
        self._drag_mode = "pan"
        self._set_zoom_mode("off")
        self._reset_zoom()

    def _init_clipboard_shortcuts(self) -> None:
        """Register shortcuts for copying the window or data region to the clipboard."""
        combos = [
            ("Ctrl+C", self._copy_full_window_to_clipboard),
            ("Meta+C", self._copy_full_window_to_clipboard),
            ("Ctrl+Shift+C", self._copy_data_region_to_clipboard),
            ("Meta+Shift+C", self._copy_data_region_to_clipboard),
        ]
        seen: set[str] = set()
        for key_str, handler in combos:
            sequence = QtGui.QKeySequence(key_str)
            portable = sequence.toString(QtGui.QKeySequence.PortableText)
            if portable in seen:
                continue
            seen.add(portable)
            shortcut = QtGui.QShortcut(sequence, self)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(handler)
            self._clipboard_shortcuts.append(shortcut)

    def _panel_colors(self) -> tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor]:
        base = self.palette().color(QtGui.QPalette.Base)
        text = self.palette().color(QtGui.QPalette.Text)
        if _color_luminance(base) < 0.45:
            bg = QtGui.QColor(base).lighter(130)
            bg.setAlpha(235)
            border = QtGui.QColor(base).lighter(170)
            border.setAlpha(200)
            return bg, border, text
        bg = QtGui.QColor(255, 255, 255, 235)
        border = QtGui.QColor(60, 60, 60, 180)
        text_color = QtGui.QColor(30, 30, 30)
        return bg, border, text_color

    def _ensure_toast(self) -> None:
        """Create a floating toast label for copy feedback."""
        if self._toast_label is not None and self._toast_opacity is not None:
            return
        label = QtWidgets.QLabel(self)
        label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        highlight = self.palette().color(QtGui.QPalette.Highlight)
        text = self.palette().color(QtGui.QPalette.HighlightedText)
        label.setStyleSheet(
            "QLabel {"
            f"background: {highlight.name()};"
            f"color: {text.name()};"
            "border-radius: 8px;"
            "padding: 6px 10px;"
            "font-weight: 600;"
            "}"
        )
        label.hide()

        opacity = QtWidgets.QGraphicsOpacityEffect(label)
        opacity.setOpacity(0.0)
        label.setGraphicsEffect(opacity)

        self._toast_label = label
        self._toast_opacity = opacity

    def _position_toast(self) -> None:
        """Place the toast near the top-right corner with a small margin."""
        if self._toast_label is None:
            return
        margin = 12
        self._toast_label.adjustSize()
        x = max(margin, self.width() - self._toast_label.width() - margin)
        y = margin
        self._toast_label.move(x, y)

    def _show_copy_notice(self, text: str) -> None:
        """Animate a short-lived toast indicating a successful action."""
        self._ensure_toast()
        if self._toast_label is None or self._toast_opacity is None:
            return

        if self._toast_animation is not None:
            self._toast_animation.stop()
            self._toast_animation.deleteLater()
            self._toast_animation = None

        self._toast_label.setText(text)
        self._position_toast()
        self._toast_label.show()
        self._toast_opacity.setOpacity(0.0)

        fade_in = QtCore.QPropertyAnimation(self._toast_opacity, b"opacity", self)
        fade_in.setDuration(150)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)

        pause = QtCore.QPauseAnimation(900, self)

        fade_out = QtCore.QPropertyAnimation(self._toast_opacity, b"opacity", self)
        fade_out.setDuration(260)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)

        group = QtCore.QSequentialAnimationGroup(self)
        group.addAnimation(fade_in)
        group.addAnimation(pause)
        group.addAnimation(fade_out)
        group.finished.connect(self._toast_label.hide)
        group.finished.connect(lambda: setattr(self, "_toast_animation", None))
        self._toast_animation = group
        group.start(QtCore.QAbstractAnimation.DeleteWhenStopped)

    def _copy_pixmap_to_clipboard(self, pixmap: QtGui.QPixmap) -> None:
        """Push a pixmap into the system clipboard if available."""
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setPixmap(pixmap)
            self._show_copy_notice("Copied")

    def _copy_full_window_to_clipboard(self) -> None:
        """Copy the full figure (axes + labels) to the clipboard as an image."""
        pixmap = self.grab()
        self._copy_pixmap_to_clipboard(pixmap)

    def _save_full_window_to_png(self) -> None:
        """Open a dialog and save the current figure snapshot as a PNG."""
        target = self.window() or self
        pixmap = self.grab()
        if pixmap.isNull():
            return

        base_name = target.windowTitle() or "imagescqt"
        default_path = f"{base_name}.png"

        # Use a persistent dialog (non-native avoids disappearing on some platforms).
        dialog = QtWidgets.QFileDialog(target, "Save window as PNG")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setNameFilter("PNG Images (*.png)")
        dialog.selectFile(default_path)
        dialog.setDefaultSuffix("png")
        dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)

        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        filename_list = dialog.selectedFiles()
        if not filename_list:
            return
        filename = filename_list[0]
        if not filename.lower().endswith(".png"):
            filename += ".png"
        if not pixmap.save(filename, "PNG"):
            QtWidgets.QMessageBox.warning(
                self,
                "Save failed",
                "Could not save the current window as a PNG file.",
            )
            return
        self._show_copy_notice("Saved PNG")

    def _copy_data_region_to_clipboard(self) -> None:
        """Copy only the data draw area to the clipboard as an image."""
        fm = QtGui.QFontMetrics(self.font())
        layout = self._current_layout or self._layout(fm)
        draw_rect: QtCore.QRect = layout.draw_rect
        if draw_rect.isEmpty():
            return
        pixmap = self.grab(draw_rect)
        self._copy_pixmap_to_clipboard(pixmap)

    def copy_axes_to_clipboard(self) -> None:
        """Copy the axes draw area to the clipboard."""
        self._copy_data_region_to_clipboard()

    def copy_figure_to_clipboard(self) -> None:
        """Copy the full figure to the clipboard."""
        self._copy_full_window_to_clipboard()

    def save_figure_to_png(self) -> None:
        """Save the full figure to a PNG image."""
        self._save_full_window_to_png()

    def _view_window(self) -> tuple[float, float, float, float]:
        """Return normalized view center and size (fractions of full image)."""
        width_frac = min(1.0, 1.0 / max(self._zoom_x, 1e-9))
        height_frac = min(1.0, 1.0 / max(self._zoom_y, 1e-9))
        half_w = width_frac / 2.0
        half_h = height_frac / 2.0
        cx = float(clip(self._view_center.x(), half_w, 1.0 - half_w))
        cy = float(clip(self._view_center.y(), half_h, 1.0 - half_h))
        self._view_center = QtCore.QPointF(cx, cy)
        return cx, cy, width_frac, height_frac

    def _apply_uniform_zoom(self, factor: float) -> None:
        """Scale zoom identically on both axes while keeping it non-negative."""
        if factor <= 0:
            return
        self._zoom_x = max(float(self._zoom_x * factor), 1.0)
        self._zoom_y = max(float(self._zoom_y * factor), 1.0)

    def _reset_zoom(self) -> None:
        """Return to the default full-view zoom and clear transient drag state."""
        self._reset_zoom_state()
        self._rubber_band = None
        self._rubber_band_kind = None
        self._dragging = False
        self._drag_target = None
        self._current_layout = None
        self._view_window()
        self.update()

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
        prepared = _prepare_image_data(data)
        qimage, vmin_resolved, vmax_resolved = _array_to_qimage(
            prepared, cmap=cmap, vmin=vmin, vmax=vmax
        )
        self._data = prepared
        self._xaxis = xaxis
        self._yaxis = yaxis
        self._cmap = cmap
        self._qimage = qimage
        self._vmin = vmin_resolved
        self._vmax = vmax_resolved
        self._colorbar_image = _make_colorbar_image(self._cmap)
        self._update_content_ratio()
        self._reset_zoom_state()
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

    def _layout(self, fm: QtGui.QFontMetrics) -> _Layout:
        """Compute layout rectangles and tick labels."""
        x_start, _, x_span = _axis_limits(self._xaxis, self._data.shape[1])
        y_start, _, y_span = _axis_limits(self._yaxis, self._data.shape[0])
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

        y_tick_block = _TICK_LEN + max_y_label_width
        margin_left = y_tick_block + pad_px
        if self._ylabel:
            margin_left += w_pad_px + max(ylabel_height, fm.height())

        tick_block = _TICK_LEN + fm.height()
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
                _COLORBAR_GAP
                + _COLORBAR_WIDTH
                + _TICK_LEN
                + max(max_cbar_label_width, cbar_scale_width)
                + max(pad_px // 2, 2)
            )
            if self._colorbar_label:
                margin_right = max(
                    margin_right,
                    _COLORBAR_GAP
                    + _COLORBAR_WIDTH
                    + _TICK_LEN
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

        return _Layout(
            tick_len=_TICK_LEN,
            cbar_width=_COLORBAR_WIDTH,
            cbar_gap=_COLORBAR_GAP,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_tick_labels=x_tick_labels,
            y_tick_labels=y_tick_labels,
            cbar_ticks=cbar_ticks,
            cbar_tick_labels=cbar_tick_labels,
            cbar_scale_text=cbar_scale_text,
            max_cbar_label_width=max_cbar_label_width,
            cbar_scale_width=cbar_scale_width,
            draw_rect=draw_rect,
            y_tick_block=y_tick_block,
            view_limits=_ViewLimits(
                x_min=x_view_min,
                x_max=x_view_max,
                y_min=y_view_min,
                y_max=y_view_max,
            ),
            view=_View(
                cx=cx,
                cy=cy,
                width_frac=width_frac,
                height_frac=height_frac,
            ),
            margins=(margin_left, margin_right, margin_top, margin_bottom),
        )

    def _pixel_from_position(
        self, pos: QtCore.QPoint, layout: _Layout
    ) -> tuple[int, int] | None:
        """Map a widget position to the nearest data pixel index."""
        draw_rect: QtCore.QRect = layout.draw_rect
        if draw_rect.width() <= 0 or draw_rect.height() <= 0:
            return None
        if not draw_rect.contains(pos):
            return None
        view_limits = layout.view_limits
        x_range = view_limits.x_max - view_limits.x_min
        y_range = view_limits.y_max - view_limits.y_min
        if x_range == 0 or y_range == 0:
            return None

        rel_x = (pos.x() - draw_rect.left()) / draw_rect.width()
        rel_y = (pos.y() - draw_rect.top()) / draw_rect.height()
        x_val = view_limits.x_min + rel_x * x_range
        y_val = view_limits.y_min + rel_y * y_range
        col_idx = _nearest_index(x_val, self._xaxis, self._data.shape[1])
        row_idx = _nearest_index(y_val, self._yaxis, self._data.shape[0])
        return col_idx, row_idx

    def _add_or_move_marker(
        self,
        pos: QtCore.QPoint,
        layout: _Layout | None = None,
        existing_idx: int | None = None,
    ) -> int | None:
        """Place marker at nearest pixel or move existing one."""
        fm = QtGui.QFontMetrics(self.font())
        layout = layout or self._current_layout or self._layout(fm)
        pixel = self._pixel_from_position(pos, layout)
        if pixel is None:
            return None
        if existing_idx is None:
            self._markers.append(_Marker(index=pixel))
            self._rubber_band = None
            self.update()
            return len(self._markers) - 1
        if not (0 <= existing_idx < len(self._markers)):
            return None
        self._markers[existing_idx].index = pixel
        self._rubber_band = None
        self.update()
        return existing_idx

    def _marker_position(
        self, marker: _Marker, layout: _Layout
    ) -> tuple[QtCore.QPointF, float, float, float] | None:
        """Return screen position plus data/axis values for a marker."""
        idx = marker.index
        if idx is None:
            return None
        col_idx, row_idx = idx
        if not (
            0 <= row_idx < self._data.shape[0]
            and 0 <= col_idx < self._data.shape[1]
        ):
            marker.index = None
            return None

        view_limits = layout.view_limits
        x_range = view_limits.x_max - view_limits.x_min
        y_range = view_limits.y_max - view_limits.y_min
        if x_range == 0 or y_range == 0:
            return None

        x_val = _axis_value(self._xaxis, col_idx)
        y_val = _axis_value(self._yaxis, row_idx)
        rel_x = (x_val - view_limits.x_min) / x_range
        rel_y = (y_val - view_limits.y_min) / y_range
        if rel_x < 0.0 or rel_x > 1.0 or rel_y < 0.0 or rel_y > 1.0:
            return None

        draw_rect: QtCore.QRect = layout.draw_rect
        px = draw_rect.left() + rel_x * draw_rect.width()
        py = draw_rect.top() + rel_y * draw_rect.height()
        data_val = float(np.abs(self._data[row_idx, col_idx]))
        return QtCore.QPointF(px, py), x_val, y_val, data_val

    def _marker_box_rect(
        self,
        marker: _Marker,
        layout: _Layout,
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
        text_height = fm.height() * len(lines)
        box_width = text_width + (_MARKER_TOOLTIP_H_PADDING * 2)
        box_height = text_height + (_MARKER_TOOLTIP_V_PADDING * 2)

        box_offset = marker.box_offset
        if box_offset is None:
            box_x = marker_pos.x() + _MARKER_BOX_OFFSET
            box_y = marker_pos.y() - _MARKER_BOX_OFFSET - box_height
            if box_x + box_width > self.width():
                box_x = marker_pos.x() - _MARKER_BOX_OFFSET - box_width
            if box_y < 0:
                box_y = marker_pos.y() + _MARKER_BOX_OFFSET
            marker.box_offset = QtCore.QPointF(
                box_x - marker_pos.x(), box_y - marker_pos.y()
            )
        else:
            box_x = marker_pos.x() + float(box_offset.x())
            box_y = marker_pos.y() + float(box_offset.y())
            box_x = float(
                clip(
                    box_x,
                    _MARKER_TOOLTIP_CLAMP_MARGIN - box_width,
                    self.width() - _MARKER_TOOLTIP_CLAMP_MARGIN,
                )
            )
            box_y = float(
                clip(
                    box_y,
                    _MARKER_TOOLTIP_CLAMP_MARGIN,
                    self.height() - _MARKER_TOOLTIP_CLAMP_MARGIN - box_height,
                )
            )
            marker.box_offset = QtCore.QPointF(
                box_x - marker_pos.x(), box_y - marker_pos.y()
            )

        box_rect = QtCore.QRectF(
            box_x,
            box_y,
            box_width,
            box_height,
        )
        return box_rect, marker_pos, lines

    def _marker_hit(
        self, pos: QtCore.QPoint, layout: _Layout
    ) -> tuple[int, _Marker] | None:
        """Check if a click is close enough to any marker handle."""
        for idx, marker in enumerate(self._markers):
            marker_info = self._marker_position(marker, layout)
            if marker_info is None:
                continue
            marker_pos, _, _, _ = marker_info
            if hypot(pos.x() - marker_pos.x(), pos.y() - marker_pos.y()) <= _MARKER_HIT_RADIUS:
                return idx, marker
        return None

    def _marker_box_hit(
        self, pos: QtCore.QPoint, layout: _Layout
    ) -> tuple[int, _Marker] | None:
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
        painter.fillRect(self.rect(), self.palette().color(QtGui.QPalette.Base))

        fm = painter.fontMetrics()
        layout = self._layout(fm)
        self._current_layout = layout
        draw_rect = layout.draw_rect
        tick_len = layout.tick_len
        cbar_width = layout.cbar_width
        cbar_gap = layout.cbar_gap
        x_ticks = layout.x_ticks
        y_ticks = layout.y_ticks
        x_tick_labels = layout.x_tick_labels
        y_tick_labels = layout.y_tick_labels
        cbar_ticks = layout.cbar_ticks
        cbar_tick_labels = layout.cbar_tick_labels
        cbar_scale_text = layout.cbar_scale_text
        max_cbar_label_width = layout.max_cbar_label_width
        cbar_scale_width = layout.cbar_scale_width
        view_limits = layout.view_limits
        view_state = layout.view
        y_tick_block = layout.y_tick_block
        margin_left, _, margin_top, margin_bottom = layout.margins

        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, self._render_smooth)
        source_rect = QtCore.QRectF(
            (view_state.cx - view_state.width_frac / 2.0) * self._qimage.width(),
            (view_state.cy - view_state.height_frac / 2.0)
            * self._qimage.height(),
            self._qimage.width() * view_state.width_frac,
            self._qimage.height() * view_state.height_frac,
        )
        painter.drawImage(QtCore.QRectF(draw_rect), self._qimage, source_rect)

        # Axes lines
        axis_pen = QtGui.QPen(self.palette().color(QtGui.QPalette.Text))
        painter.setPen(axis_pen)
        painter.drawLine(
            draw_rect.left(), draw_rect.top(), draw_rect.left(), draw_rect.bottom()
        )  # y axis
        painter.drawLine(
            draw_rect.left(), draw_rect.bottom(), draw_rect.right(), draw_rect.bottom()
        )  # x axis

        # X ticks
        x_range = view_limits.x_max - view_limits.x_min
        for val, label in zip(x_ticks, x_tick_labels):
            pos = (
                draw_rect.left()
                if x_range == 0
                else draw_rect.left()
                + int((val - view_limits.x_min) / x_range * draw_rect.width())
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
        y_range = view_limits.y_max - view_limits.y_min
        for val, label in zip(y_ticks, y_tick_labels):
            pos = (
                draw_rect.top()
                if y_range == 0
                else draw_rect.top()
                + int((val - view_limits.y_min) / y_range * draw_rect.height())
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
                QtCore.QRectF(0, 0, self.width(), layout.margins[2]),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignBottom,
                self._title,
            )
        self._draw_marker(painter, layout)
        self._draw_rubber_band(painter)
        self._schedule_auto_shrink()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        """Keep toast anchored near the corner on resize."""
        super().resizeEvent(event)
        self._position_toast()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        """Zoom in/out with mouse wheel."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        step = 1.1 if delta > 0 else 0.9
        if self._zoom_axis_mode == "x":
            self._zoom_x = max(float(self._zoom_x * step), 1.0)
        elif self._zoom_axis_mode == "y":
            self._zoom_y = max(float(self._zoom_y * step), 1.0)
        else:
            self._apply_uniform_zoom(step)
        self._current_layout = None
        self._view_window()  # clamp center after zoom changes
        self.update()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        """Reset zoom when the user double-clicks with the left mouse button."""
        if event.button() == QtCore.Qt.LeftButton:
            self._reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

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
                marker.box_offset = QtCore.QPointF(
                    box_rect.left() - marker_pos.x(),
                    box_rect.top() - marker_pos.y(),
                )
            event.accept()
            return

        if self._zoom_axis_mode != "off":
            self._drag_target = None
            self._dragging = True
            self._drag_start = event.pos()
            self._view_center_start = QtCore.QPointF(self._view_center)
            self._rubber_band = None
            self._rubber_band_kind = None
            self._zoom_alt = bool(event.modifiers() & QtCore.Qt.AltModifier)
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
                    current_offset: QtCore.QPointF = marker.box_offset or QtCore.QPointF(0, 0)
                    marker.box_offset = QtCore.QPointF(
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
        if self._zoom_axis_mode != "off":
            fm = QtGui.QFontMetrics(self.font())
            layout = self._current_layout or self._layout(fm)
            zoom_rect = self._zoom_selection_rect(
                layout.draw_rect, self._drag_start, event.pos(), self._zoom_axis_mode
            )
            self._rubber_band = zoom_rect
            self._rubber_band_kind = (
                f"zoom-{self._zoom_axis_mode}" if zoom_rect is not None else None
            )
            self.update()
            event.accept()
            return
        if self._drag_mode == "pan":
            delta = event.pos() - self._drag_start
            layout = self._current_layout or self._layout(
                QtGui.QFontMetrics(self.font())
            )
            draw_rect: QtCore.QRect = layout.draw_rect
            if draw_rect.width() > 0 and draw_rect.height() > 0:
                view_state = layout.view
                width_frac = view_state.width_frac
                height_frac = view_state.height_frac
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
            self._rubber_band_kind = "box"
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
        if self._zoom_axis_mode != "off":
            self._dragging = False
            applied = False
            move_x = abs(event.pos().x() - self._drag_start.x())
            move_y = abs(event.pos().y() - self._drag_start.y())
            threshold = 3
            if self._zoom_axis_mode == "x":
                has_band = move_x > threshold
            elif self._zoom_axis_mode == "y":
                has_band = move_y > threshold
            else:
                has_band = move_x > threshold or move_y > threshold
            if has_band and self._rubber_band is not None:
                applied = self._apply_zoom_selection(
                    self._rubber_band,
                    self._zoom_axis_mode,
                    zoom_out=self._zoom_alt,
                )
            else:
                applied = self._apply_click_zoom(
                    self._drag_start,
                    self._zoom_axis_mode,
                    zoom_out=self._zoom_alt,
                )
            self._rubber_band = None
            self._rubber_band_kind = None
            self._current_layout = None
            self._zoom_alt = False
            if applied:
                event.accept()
                return
        self._dragging = False
        if self._drag_mode == "box" and self._rubber_band is not None:
            self._apply_box_zoom(self._rubber_band)
            self._rubber_band = None
            self._rubber_band_kind = None
        self._current_layout = None
        self.update()
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == QtCore.Qt.Key_S and (
            event.modifiers() & QtCore.Qt.ShiftModifier
        ):
            self._drag_mode = "box" if self._drag_mode == "pan" else "pan"
            self._set_zoom_mode("off")
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
        def _menu_text(label: str, shortcut: str | None = None) -> str:
            if not shortcut:
                return label
            sequence = QtGui.QKeySequence(shortcut)
            return f"{label} ({sequence.toString(QtGui.QKeySequence.NativeText)})"

        fm = QtGui.QFontMetrics(self.font())
        layout = self._current_layout or self._layout(fm)
        menu = QtWidgets.QMenu(self)
        zoom_action = menu.addAction(_menu_text("Zoom", "Z"))
        zoom_x_action = menu.addAction(_menu_text("Zoom X", "H"))
        zoom_y_action = menu.addAction(_menu_text("Zoom Y", "V"))
        pan_action = menu.addAction("Pan mode")
        box_action = menu.addAction("Box-zoom mode")
        mode_group = QtGui.QActionGroup(menu)
        for act in (zoom_action, zoom_x_action, zoom_y_action, pan_action, box_action):
            act.setCheckable(True)
            mode_group.addAction(act)
        mode_group.setExclusive(True)
        menu.addSeparator()
        copy_window_action = menu.addAction(
            _menu_text("Copy figure to clipboard", "Y")
        )
        copy_data_action = menu.addAction(
            _menu_text("Copy axes area to clipboard", "Shift+Y")
        )
        save_png_action = menu.addAction(_menu_text("Save figure as PNG...", "S"))
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
        current_zoom = self._zoom_axis_mode
        if current_zoom == "xy":
            zoom_action.setChecked(True)
        elif current_zoom == "x":
            zoom_x_action.setChecked(True)
        elif current_zoom == "y":
            zoom_y_action.setChecked(True)
        elif self._drag_mode == "pan":
            pan_action.setChecked(True)
        else:
            box_action.setChecked(True)

        chosen = menu.exec(self.mapToGlobal(pos))
        if chosen == zoom_action:
            self._set_zoom_mode("xy")
        elif chosen == zoom_x_action:
            self._set_zoom_mode("x")
        elif chosen == zoom_y_action:
            self._set_zoom_mode("y")
        elif chosen == pan_action:
            self._set_zoom_mode("off")
            self._drag_mode = "pan"
        elif chosen == box_action:
            self._set_zoom_mode("off")
            self._drag_mode = "box"
        elif chosen == copy_window_action:
            self._copy_full_window_to_clipboard()
        elif chosen == copy_data_action:
            self._copy_data_region_to_clipboard()
        elif chosen == save_png_action:
            self._save_full_window_to_png()
        elif delete_action is not None and chosen == delete_action and target_idx is not None:
            self._clear_marker(target_idx)

    def _zoom_selection_rect(
        self,
        draw_rect: QtCore.QRect,
        start: QtCore.QPoint,
        current: QtCore.QPoint,
        axis_mode: str,
    ) -> QtCore.QRect | None:
        """Clamp and shape the rubber-band rectangle for zoom drag."""
        if draw_rect.isEmpty():
            return None
        start_x = max(min(start.x(), draw_rect.right()), draw_rect.left())
        start_y = max(min(start.y(), draw_rect.bottom()), draw_rect.top())
        cur_x = max(min(current.x(), draw_rect.right()), draw_rect.left())
        cur_y = max(min(current.y(), draw_rect.bottom()), draw_rect.top())

        if axis_mode == "x":
            left = min(start_x, cur_x)
            right = max(start_x, cur_x)
            return QtCore.QRect(
                QtCore.QPoint(left, draw_rect.top()),
                QtCore.QPoint(right, draw_rect.bottom()),
            ).normalized()
        if axis_mode == "y":
            top = min(start_y, cur_y)
            bottom = max(start_y, cur_y)
            return QtCore.QRect(
                QtCore.QPoint(draw_rect.left(), top),
                QtCore.QPoint(draw_rect.right(), bottom),
            ).normalized()

        rect = QtCore.QRect(
            QtCore.QPoint(start_x, start_y), QtCore.QPoint(cur_x, cur_y)
        ).normalized()
        rect = rect & draw_rect
        if rect.isEmpty():
            return None
        return rect

    def _apply_click_zoom(
        self,
        pos: QtCore.QPoint,
        axis_mode: str,
        *,
        zoom_out: bool = False,
    ) -> bool:
        """Zoom toward the clicked position along the requested axis mode."""
        fm = QtGui.QFontMetrics(self.font())
        layout = self._current_layout or self._layout(fm)
        draw_rect: QtCore.QRect = layout.draw_rect
        if draw_rect.isEmpty() or not draw_rect.contains(pos):
            return False

        view_state = layout.view
        draw_w = max(draw_rect.width(), 1)
        draw_h = max(draw_rect.height(), 1)
        rel_x = (pos.x() - draw_rect.left()) / draw_w
        rel_y = (pos.y() - draw_rect.top()) / draw_h
        view_left = view_state.cx - view_state.width_frac / 2.0
        view_top = view_state.cy - view_state.height_frac / 2.0
        self._view_center = QtCore.QPointF(
            view_left + rel_x * view_state.width_frac,
            view_top + rel_y * view_state.height_frac,
        )

        factor = 1.1
        if zoom_out:
            factor = 1.0 / factor
        if axis_mode in {"xy", "x"}:
            self._zoom_x = max(float(self._zoom_x * factor), 1.0)
        if axis_mode in {"xy", "y"}:
            self._zoom_y = max(float(self._zoom_y * factor), 1.0)

        self._view_window()
        self._current_layout = None
        self.update()
        return True

    def _apply_zoom_selection(
        self,
        selection: QtCore.QRect,
        axis_mode: str,
        *,
        zoom_out: bool = False,
    ) -> bool:
        """Apply drag-based zoom according to the current zoom axis mode."""
        fm = QtGui.QFontMetrics(self.font())
        layout = self._current_layout or self._layout(fm)
        draw_rect: QtCore.QRect = layout.draw_rect
        intersect = selection & draw_rect
        if intersect.isEmpty():
            return False

        sel_w = max(intersect.width(), 1)
        sel_h = max(intersect.height(), 1)
        draw_w = max(draw_rect.width(), 1)
        draw_h = max(draw_rect.height(), 1)

        view_state = layout.view
        center_norm_x = (intersect.center().x() - draw_rect.left()) / draw_w
        center_norm_y = (intersect.center().y() - draw_rect.top()) / draw_h
        view_left = view_state.cx - view_state.width_frac / 2.0
        view_top = view_state.cy - view_state.height_frac / 2.0
        new_center = QtCore.QPointF(
            view_left + center_norm_x * view_state.width_frac,
            view_top + center_norm_y * view_state.height_frac,
        )

        factor_w = draw_w / sel_w
        factor_h = draw_h / sel_h
        if zoom_out:
            factor_w = 1.0 / factor_w
            factor_h = 1.0 / factor_h

        cx = self._view_center.x()
        cy = self._view_center.y()
        if axis_mode in {"xy", "x"}:
            cx = new_center.x()
        if axis_mode in {"xy", "y"}:
            cy = new_center.y()
        self._view_center = QtCore.QPointF(cx, cy)

        if axis_mode == "xy":
            if self._aspect_mode == "equal":
                self._apply_uniform_zoom(min(factor_w, factor_h))
            else:
                self._zoom_x = max(float(self._zoom_x * factor_w), 1.0)
                self._zoom_y = max(float(self._zoom_y * factor_h), 1.0)
        elif axis_mode == "x":
            self._zoom_x = max(float(self._zoom_x * factor_w), 1.0)
        elif axis_mode == "y":
            self._zoom_y = max(float(self._zoom_y * factor_h), 1.0)
        else:
            return False

        self._view_window()
        self._current_layout = None
        self.update()
        return True

    def _apply_box_zoom(self, selection: QtCore.QRect) -> None:
        """Zoom into the selected rectangle."""
        fm = QtGui.QFontMetrics(self.font())
        layout = self._current_layout or self._layout(fm)
        draw_rect: QtCore.QRect = layout.draw_rect
        intersect = selection & draw_rect
        if intersect.isEmpty():
            return

        sel_w = max(intersect.width(), 1)
        sel_h = max(intersect.height(), 1)
        draw_w = max(draw_rect.width(), 1)
        draw_h = max(draw_rect.height(), 1)

        center_norm_x = (intersect.center().x() - draw_rect.left()) / draw_w
        center_norm_y = (intersect.center().y() - draw_rect.top()) / draw_h
        view_state = layout.view
        view_left = view_state.cx - view_state.width_frac / 2.0
        view_top = view_state.cy - view_state.height_frac / 2.0
        new_center = QtCore.QPointF(
            view_left + center_norm_x * view_state.width_frac,
            view_top + center_norm_y * view_state.height_frac,
        )

        factor_w = draw_w / sel_w
        factor_h = draw_h / sel_h

        self._view_center = new_center
        if self._aspect_mode == "equal":
            self._apply_uniform_zoom(min(factor_w, factor_h))
        else:
            # Stretch zoom per-axis so the rubber-band region fills the draw area.
            self._zoom_x = max(float(self._zoom_x * factor_w), 1.0)
            self._zoom_y = max(float(self._zoom_y * factor_h), 1.0)
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

    def _draw_marker(self, painter: QtGui.QPainter, layout: _Layout) -> None:
        """Render MATLAB-style data markers and tooltips."""
        if not self._markers:
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        fm = painter.fontMetrics()
        highlight = QtGui.QColor("#d32f2f")
        panel_bg, panel_border, panel_text = self._panel_colors()

        for marker in self._markers:
            box_info = self._marker_box_rect(marker, layout, fm)
            if box_info is None:
                continue
            box_rect, marker_pos, lines = box_info

            outer_pen = QtGui.QPen(QtGui.QColor("white"))
            outer_pen.setWidth(_MARKER_OUTER_WIDTH)
            painter.setPen(outer_pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(marker_pos, _MARKER_RADIUS_OUTER, _MARKER_RADIUS_OUTER)

            inner_pen = QtGui.QPen(highlight)
            inner_pen.setWidth(_MARKER_INNER_WIDTH)
            painter.setPen(inner_pen)
            painter.drawEllipse(marker_pos, _MARKER_RADIUS_INNER, _MARKER_RADIUS_INNER)
            painter.drawLine(
                marker_pos + QtCore.QPointF(-_MARKER_CROSS_HALF, 0),
                marker_pos + QtCore.QPointF(_MARKER_CROSS_HALF, 0),
            )
            painter.drawLine(
                marker_pos + QtCore.QPointF(0, -_MARKER_CROSS_HALF),
                marker_pos + QtCore.QPointF(0, _MARKER_CROSS_HALF),
            )

            painter.setPen(QtGui.QPen(highlight, 1.5))
            anchor_point = QtCore.QPointF(
                box_rect.left() + _MARKER_ANCHOR_OFFSET,
                box_rect.top() + box_rect.height() / 2,
            )
            painter.drawLine(marker_pos, anchor_point)

            painter.setBrush(panel_bg)
            painter.setPen(QtGui.QPen(panel_border, 1))
            painter.drawRoundedRect(
                box_rect,
                _MARKER_TOOLTIP_CORNER_RADIUS,
                _MARKER_TOOLTIP_CORNER_RADIUS,
            )

            text_y = box_rect.top() + _MARKER_TOOLTIP_V_PADDING + fm.ascent()
            painter.setPen(QtGui.QPen(panel_text))
            for line in lines:
                painter.drawText(
                    box_rect.left() + _MARKER_TOOLTIP_H_PADDING, text_y, line
                )
                text_y += fm.height()

        painter.restore()

    def _draw_rubber_band(self, painter: QtGui.QPainter) -> None:
        if self._rubber_band is None:
            return
        pen_color, fill_color = _rubber_band_colors(self._cmap)
        pen = QtGui.QPen(pen_color)
        pen.setStyle(QtCore.Qt.DashLine)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(fill_color))
        painter.drawRect(self._rubber_band)

    def _schedule_auto_shrink(self) -> None:
        """Avoid resizing during paint; defer with a zero-timeout timer."""
        if not (self._tight_enabled and self._tight_auto_resize_pending):
            return
        if self._auto_resize_timer is None:
            self._auto_resize_timer = QtCore.QTimer(self)
            self._auto_resize_timer.setSingleShot(True)
            self._auto_resize_timer.timeout.connect(self._auto_shrink_window)
        self._auto_resize_timer.start(0)

    def _auto_shrink_window(self) -> None:
        """Optionally shrink the parent window to fit content more tightly."""
        if not (self._tight_enabled and self._tight_auto_resize_pending):
            return
        fm = QtGui.QFontMetrics(self.font())
        layout = self._current_layout or self._layout(fm)
        if layout is None:
            return
        win = self.window()
        if win is None:
            return

        margins = layout.margins
        draw_rect: QtCore.QRect = layout.draw_rect
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
        aspect: str = "auto",
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
        self._toolbar = FigureToolbar(self._canvas, parent=self)
        self._toolbar.setObjectName("qtplotlibToolbar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self._toolbar)
        self.resize(720, 540)
        self._theme = _THEME_DARK
        self.set_theme(self._theme)

    def theme(self) -> str:
        """Return the active theme name."""
        return self._theme

    def set_theme(self, theme: str) -> None:
        """Apply a light or dark theme to the window."""
        self._theme = _apply_theme_to_widget(self, theme)
        self._canvas.update()

    def toggle_theme(self) -> None:
        """Toggle between light and dark themes."""
        next_theme = _THEME_LIGHT if self._theme == _THEME_DARK else _THEME_DARK
        self.set_theme(next_theme)

    def set_image(
        self,
        *args: object,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        xaxis: ndarray | None = None,
        yaxis: ndarray | None = None,
    ) -> None:
        """Update the displayed image and optional axes (data or x, y, data)."""
        data_arg, xaxis_arg, yaxis_arg = _parse_axes_and_data(
            args, xaxis, yaxis, "set_image"
        )
        self._canvas.set_image(
            data_arg, cmap=cmap, vmin=vmin, vmax=vmax, xaxis=xaxis_arg, yaxis=yaxis_arg
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
    *args: object,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "imagescqt",
    interpolation: str = "nearest",
    aspect: str = "equal",
    xlabel: str = "",
    ylabel: str = "",
    colorbar: bool = False,
    colorbar_label: str = "",
    xaxis: ndarray | None = None,
    yaxis: ndarray | None = None,
) -> ImageWindow:
    """
    Display a numpy array in a PySide6 window (MATLAB imagesc style).

    When an application instance already exists (e.g. IPython with "%gui qt"),
    the window is shown and control returns immediately. If no application is
    running, a new one is created and the call will block until the window
    closes.
    """
    data_arg, xaxis_arg, yaxis_arg = _parse_axes_and_data(
        args, xaxis, yaxis, "imagescqt"
    )
    arr = asarray(data_arg)
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
        xaxis=xaxis_arg,
        yaxis=yaxis_arg,
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
