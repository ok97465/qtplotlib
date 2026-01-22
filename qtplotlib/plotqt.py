"""Lightweight PySide6 line plotting helper with a Matplotlib-like toolbar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy import asarray, isfinite, ndarray

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:
    raise ImportError("PySide6 is required to use plotqt.") from exc

from .imagescqt import FigureToolbar

_TICK_LEN = 6
_DEFAULT_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)
_COLOR_ALIASES = {
    "b": "#1f77b4",
    "g": "#2ca02c",
    "r": "#d62728",
    "c": "#17becf",
    "m": "#e377c2",
    "y": "#bcbd22",
    "k": "#000000",
    "w": "#ffffff",
}
_MARKER_SET = {"o", "s", "^", "v", "x", "+", "*", ".", "D"}
_LINESTYLES = ("--", "-.", ":", "-")
_DATA_MARKER_OUTER_WIDTH = 3
_DATA_MARKER_INNER_WIDTH = 2
_DATA_MARKER_RADIUS_OUTER = 6
_DATA_MARKER_RADIUS_INNER = 5
_DATA_MARKER_CROSS_HALF = 8
_DATA_MARKER_HIT_RADIUS = 8.0
_DATA_MARKER_BOX_OFFSET = 10
_DATA_MARKER_TOOLTIP_H_PADDING = 6
_DATA_MARKER_TOOLTIP_V_PADDING = 4
_DATA_MARKER_TOOLTIP_CLAMP_MARGIN = 4.0
_DATA_MARKER_TOOLTIP_CORNER_RADIUS = 6
_DATA_MARKER_ANCHOR_OFFSET = 6


@dataclass(frozen=True)
class _PlotStyle:
    color: str | QtGui.QColor | None
    linestyle: str | None
    marker: str | None


@dataclass
class _PlotSeries:
    x: ndarray
    y: ndarray
    label: str
    color: QtGui.QColor
    linestyle: str
    marker: str | None
    linewidth: float
    markersize: float
    alpha: float


@dataclass(frozen=True)
class _PlotLayout:
    draw_rect: QtCore.QRect
    x_ticks: list[float]
    y_ticks: list[float]
    x_labels: list[str]
    y_labels: list[str]
    margins: tuple[int, int, int, int]


@dataclass
class _PlotMarker:
    series_idx: int
    point_idx: int
    box_offset: QtCore.QPointF | None = None


def _coerce_1d(data: object, name: str) -> ndarray:
    arr = asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence.")
    return arr


def _parse_plot_args(
    args: tuple[object, ...],
) -> list[tuple[object | None, object, str | None]]:
    if not args:
        raise TypeError("plotqt requires data arguments.")

    series: list[tuple[object | None, object, str | None]] = []
    idx = 0
    while idx < len(args):
        if isinstance(args[idx], str):
            raise TypeError("Format string provided without data.")
        if idx + 1 < len(args) and not isinstance(args[idx + 1], str):
            x_val = args[idx]
            y_val = args[idx + 1]
            idx += 2
        else:
            x_val = None
            y_val = args[idx]
            idx += 1

        fmt = None
        if idx < len(args) and isinstance(args[idx], str):
            fmt = args[idx]
            idx += 1
        series.append((x_val, y_val, fmt))
    return series


def _parse_format_string(fmt: str | None) -> _PlotStyle:
    if not fmt:
        return _PlotStyle(color=None, linestyle=None, marker=None)
    fmt = fmt.strip()
    if not fmt or fmt.lower() == "none":
        return _PlotStyle(color=None, linestyle=None, marker=None)

    color: str | None = None
    linestyle: str | None = None
    marker: str | None = None

    for char in fmt:
        if char in _COLOR_ALIASES:
            color = _COLOR_ALIASES[char]
            break

    for style in _LINESTYLES:
        if style in fmt:
            linestyle = style
            break

    for char in fmt:
        if char in _MARKER_SET:
            marker = char
            break

    return _PlotStyle(color=color, linestyle=linestyle, marker=marker)


def _format_tick(val: float) -> str:
    return f"{val:.4g}"


def _tick_values(min_val: float, max_val: float, count: int = 5) -> list[float]:
    if count <= 1:
        return [min_val]
    if min_val == max_val:
        return [min_val]
    step = (max_val - min_val) / (count - 1)
    return [min_val + step * i for i in range(count)]


def _to_qcolor(color: str | QtGui.QColor | None) -> QtGui.QColor:
    if isinstance(color, QtGui.QColor):
        return color
    qcolor = QtGui.QColor(str(color)) if color is not None else QtGui.QColor()
    if not qcolor.isValid():
        return QtGui.QColor("#111111")
    return qcolor


def _coerce_sequence(value: object, count: int) -> list[object | None]:
    if value is None:
        return [None] * count
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != count:
            raise ValueError("Style sequence length must match number of series.")
        return list(value)
    return [value] * count


def _color_luminance(color: QtGui.QColor) -> float:
    r, g, b, _ = color.getRgbF()
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


class _PlotCanvas(QtWidgets.QWidget):
    """Custom canvas that draws line plots with axes, grid, and legend."""

    zoomModeChanged = QtCore.Signal(str)

    def __init__(
        self,
        *,
        title: str,
        xlabel: str,
        ylabel: str,
        grid: bool,
        legend: bool,
        xlim: tuple[float, float] | None,
        ylim: tuple[float, float] | None,
    ) -> None:
        super().__init__()
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._grid = bool(grid)
        self._legend = bool(legend)
        self._series: list[_PlotSeries] = []
        self._color_index = 0
        self._last_layout: _PlotLayout | None = None

        self._data_xlim = (0.0, 1.0)
        self._data_ylim = (0.0, 1.0)
        self._manual_xlim = xlim is not None
        self._manual_ylim = ylim is not None
        self._home_xlim = xlim if xlim is not None else self._data_xlim
        self._home_ylim = ylim if ylim is not None else self._data_ylim
        self._view_xlim = self._home_xlim
        self._view_ylim = self._home_ylim

        self._zoom_axis_mode = "off"
        self._dragging = False
        self._drag_start = QtCore.QPoint()
        self._pan_xlim_start = self._view_xlim
        self._pan_ylim_start = self._view_ylim
        self._rubber_band: QtCore.QRect | None = None
        self._tight_enabled = False
        self._tight_pad = 1.08
        self._markers: list[_PlotMarker] = []
        self._drag_target: dict[str, object] | None = None
        self._box_drag_start = QtCore.QPoint()
        self._toast_label: QtWidgets.QLabel | None = None
        self._toast_timer: QtCore.QTimer | None = None

        self.setMinimumSize(360, 260)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    @property
    def zoom_axis_mode(self) -> str:
        """Return the current zoom mode (off, xy, x, y)."""
        return self._zoom_axis_mode

    def set_zoom_axis_mode(self, mode: str) -> None:
        """Set the active zoom axis mode."""
        if mode not in {"off", "xy", "x", "y"}:
            mode = "off"
        self._zoom_axis_mode = mode
        self._dragging = False
        self._rubber_band = None
        self._update_zoom_cursor()
        self.zoomModeChanged.emit(self._zoom_axis_mode)
        self.update()

    def reset_view(self) -> None:
        """Reset the plot view to the home limits."""
        self._view_xlim = self._home_xlim
        self._view_ylim = self._home_ylim
        self._rubber_band = None
        self._dragging = False
        self.update()

    def set_title(self, text: str) -> None:
        self._title = text
        self.update()

    def set_xlabel(self, text: str) -> None:
        self._xlabel = text
        self.update()

    def set_ylabel(self, text: str) -> None:
        self._ylabel = text
        self.update()

    def set_grid(self, enabled: bool) -> None:
        self._grid = bool(enabled)
        self.update()

    def set_legend(self, enabled: bool) -> None:
        self._legend = bool(enabled)
        self.update()

    def set_xlim(self, xlim: tuple[float, float]) -> None:
        self._manual_xlim = True
        self._home_xlim = xlim
        self._view_xlim = xlim
        self.update()

    def set_ylim(self, ylim: tuple[float, float]) -> None:
        self._manual_ylim = True
        self._home_ylim = ylim
        self._view_ylim = ylim
        self.update()

    def clear(self) -> None:
        """Remove all plotted series."""
        self._series.clear()
        self._color_index = 0
        self._markers.clear()
        self._update_limits(reset_view=True)
        self.update()

    def tight_layout(self, *, pad: float = 1.08) -> None:
        """Use compact margins similar to matplotlib's tight_layout."""
        self._tight_enabled = True
        self._tight_pad = float(pad)
        self.update()

    def disable_tight_layout(self) -> None:
        """Disable tight_layout margins."""
        self._tight_enabled = False
        self.update()

    def set_series(self, series: Iterable[_PlotSeries]) -> None:
        """Replace all series with a new collection."""
        self._series = list(series)
        self._color_index = len(self._series)
        self._markers.clear()
        self._update_limits(reset_view=True)
        self.update()

    def add_series(self, series: Iterable[_PlotSeries]) -> None:
        """Append new series to the plot without resetting the view."""
        self._series.extend(series)
        self._color_index = len(self._series)
        self._update_limits(reset_view=False)
        self.update()

    def copy_axes_to_clipboard(self) -> None:
        """Copy only the axes region to the clipboard."""
        layout = self._last_layout
        if layout is None or layout.draw_rect.isEmpty():
            return
        pixmap = self.grab(layout.draw_rect)
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setPixmap(pixmap)
            self._show_copy_notice("Copied")

    def copy_figure_to_clipboard(self) -> None:
        """Copy the full figure window to the clipboard."""
        pixmap = self.grab()
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setPixmap(pixmap)
            self._show_copy_notice("Copied")

    def save_figure_to_png(self) -> None:
        """Save the full figure to a PNG file."""
        target = self.window() or self
        pixmap = self.grab()
        if pixmap.isNull():
            return

        base_name = target.windowTitle() or "plotqt"
        default_path = f"{base_name}.png"
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

    def _update_limits(self, *, reset_view: bool) -> None:
        x_min, x_max, y_min, y_max = self._compute_data_limits()
        self._data_xlim = (x_min, x_max)
        self._data_ylim = (y_min, y_max)

        if not self._manual_xlim:
            self._home_xlim = self._data_xlim
            if reset_view:
                self._view_xlim = self._home_xlim
        if not self._manual_ylim:
            self._home_ylim = self._data_ylim
            if reset_view:
                self._view_ylim = self._home_ylim

    def _ensure_toast(self) -> None:
        if self._toast_label is not None and self._toast_timer is not None:
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
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(label.hide)
        self._toast_label = label
        self._toast_timer = timer

    def _position_toast(self) -> None:
        if self._toast_label is None:
            return
        margin = 12
        self._toast_label.adjustSize()
        x = max(margin, self.width() - self._toast_label.width() - margin)
        y = margin
        self._toast_label.move(x, y)

    def _show_copy_notice(self, text: str) -> None:
        self._ensure_toast()
        if self._toast_label is None or self._toast_timer is None:
            return
        self._toast_label.setText(text)
        self._position_toast()
        self._toast_label.show()
        self._toast_timer.start(900)

    def _compute_data_limits(self) -> tuple[float, float, float, float]:
        if not self._series:
            return 0.0, 1.0, 0.0, 1.0

        x_vals: list[float] = []
        y_vals: list[float] = []
        for series in self._series:
            mask = isfinite(series.x) & isfinite(series.y)
            if not mask.any():
                continue
            x_vals.append(float(series.x[mask].min()))
            x_vals.append(float(series.x[mask].max()))
            y_vals.append(float(series.y[mask].min()))
            y_vals.append(float(series.y[mask].max()))

        if not x_vals or not y_vals:
            return 0.0, 1.0, 0.0, 1.0

        x_min = min(x_vals)
        x_max = max(x_vals)
        y_min = min(y_vals)
        y_max = max(y_vals)

        x_min, x_max = _pad_limits(x_min, x_max)
        y_min, y_max = _pad_limits(y_min, y_max)
        return x_min, x_max, y_min, y_max

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

    def _update_zoom_cursor(self) -> None:
        if self._zoom_axis_mode == "xy":
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        elif self._zoom_axis_mode == "x":
            self.setCursor(QtGui.QCursor(QtCore.Qt.SizeHorCursor))
        elif self._zoom_axis_mode == "y":
            self.setCursor(QtGui.QCursor(QtCore.Qt.SizeVerCursor))
        else:
            self.unsetCursor()

    def _layout(self, fm: QtGui.QFontMetrics) -> _PlotLayout:
        x_min, x_max = self._view_xlim
        y_min, y_max = self._view_ylim
        x_ticks = _tick_values(x_min, x_max)
        y_ticks = _tick_values(y_min, y_max)
        x_labels = [_format_tick(val) for val in x_ticks]
        y_labels = [_format_tick(val) for val in y_ticks]

        y_label_width = max((fm.horizontalAdvance(label) for label in y_labels), default=0)
        x_label_height = fm.height()
        title_height = fm.height() if self._title else 0
        xlabel_height = fm.height() if self._xlabel else 0
        ylabel_width = fm.horizontalAdvance(self._ylabel) if self._ylabel else 0

        base_pad = self._tight_pad if self._tight_enabled else 1.8
        left = y_label_width + _TICK_LEN + int(6 * base_pad)
        if self._ylabel:
            left += ylabel_width + int(4 * base_pad)
        bottom = _TICK_LEN + x_label_height + int(6 * base_pad)
        if self._xlabel:
            bottom += xlabel_height + int(3 * base_pad)
        top = title_height + int(6 * base_pad)
        right = int(8 * base_pad)

        draw_rect = QtCore.QRect(
            left,
            top,
            max(1, self.width() - left - right),
            max(1, self.height() - top - bottom),
        )
        margins = (left, right, top, bottom)
        return _PlotLayout(
            draw_rect=draw_rect,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            x_labels=x_labels,
            y_labels=y_labels,
            margins=margins,
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        fm = painter.fontMetrics()
        layout = self._layout(fm)
        self._last_layout = layout

        draw_rect = layout.draw_rect
        axis_color = self.palette().color(QtGui.QPalette.Text)
        grid_color = QtGui.QColor(axis_color)
        grid_color.setAlpha(50)

        painter.fillRect(self.rect(), self.palette().color(QtGui.QPalette.Base))

        painter.save()
        painter.setClipRect(draw_rect)
        if self._grid:
            self._draw_grid(painter, layout, grid_color)
        self._draw_series(painter, layout)
        painter.restore()

        painter.setPen(QtGui.QPen(axis_color, 1))
        painter.drawRect(draw_rect)
        self._draw_ticks_and_labels(painter, layout, fm, axis_color)
        self._draw_titles(painter, layout, fm, axis_color)

        if self._legend:
            self._draw_legend(painter, layout, fm)

        self._draw_markers(painter, layout, fm)

        if self._rubber_band is not None:
            pen = QtGui.QPen(QtGui.QColor(40, 40, 40), 1, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtGui.QColor(40, 40, 40, 40))
            painter.drawRect(self._rubber_band)

    def _draw_grid(
        self,
        painter: QtGui.QPainter,
        layout: _PlotLayout,
        grid_color: QtGui.QColor,
    ) -> None:
        painter.setPen(QtGui.QPen(grid_color, 1, QtCore.Qt.DotLine))
        x_min, x_max = self._view_xlim
        y_min, y_max = self._view_ylim
        rect = layout.draw_rect
        x_span = max(x_max - x_min, 1e-12)
        y_span = max(y_max - y_min, 1e-12)
        for value in layout.x_ticks:
            x_pos = rect.left() + (value - x_min) / x_span * rect.width()
            painter.drawLine(int(x_pos), rect.top(), int(x_pos), rect.bottom())
        for value in layout.y_ticks:
            y_pos = rect.bottom() - (value - y_min) / y_span * rect.height()
            painter.drawLine(rect.left(), int(y_pos), rect.right(), int(y_pos))

    def _draw_series(self, painter: QtGui.QPainter, layout: _PlotLayout) -> None:
        rect = layout.draw_rect
        if rect.isEmpty():
            return
        x_min, x_max = self._view_xlim
        y_min, y_max = self._view_ylim
        x_span = max(x_max - x_min, 1e-12)
        y_span = max(y_max - y_min, 1e-12)

        for series in self._series:
            pen = QtGui.QPen(series.color, series.linewidth)
            pen.setStyle(_linestyle_to_qt(series.linestyle))
            pen.setCosmetic(True)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setOpacity(max(0.0, min(series.alpha, 1.0)))

            points: list[QtCore.QPointF] = []
            for x_val, y_val in zip(series.x, series.y, strict=False):
                if not (isfinite(x_val) and isfinite(y_val)):
                    _flush_line(painter, points)
                    points.clear()
                    continue
                x_pos = rect.left() + (x_val - x_min) / x_span * rect.width()
                y_pos = rect.bottom() - (y_val - y_min) / y_span * rect.height()
                points.append(QtCore.QPointF(x_pos, y_pos))
            _flush_line(painter, points)

            if series.marker:
                painter.setOpacity(max(0.0, min(series.alpha, 1.0)))
                painter.setPen(QtGui.QPen(series.color, max(1.0, series.linewidth)))
                painter.setBrush(series.color)
                for x_val, y_val in zip(series.x, series.y, strict=False):
                    if not (isfinite(x_val) and isfinite(y_val)):
                        continue
                    x_pos = rect.left() + (x_val - x_min) / x_span * rect.width()
                    y_pos = rect.bottom() - (y_val - y_min) / y_span * rect.height()
                    _draw_marker(
                        painter,
                        QtCore.QPointF(x_pos, y_pos),
                        series.marker,
                        series.markersize,
                    )

        painter.setOpacity(1.0)

    def _draw_ticks_and_labels(
        self,
        painter: QtGui.QPainter,
        layout: _PlotLayout,
        fm: QtGui.QFontMetrics,
        axis_color: QtGui.QColor,
    ) -> None:
        rect = layout.draw_rect
        x_min, x_max = self._view_xlim
        y_min, y_max = self._view_ylim
        x_span = max(x_max - x_min, 1e-12)
        y_span = max(y_max - y_min, 1e-12)

        painter.setPen(QtGui.QPen(axis_color, 1))
        for value, label in zip(layout.x_ticks, layout.x_labels, strict=False):
            x_pos = rect.left() + (value - x_min) / x_span * rect.width()
            painter.drawLine(
                QtCore.QPointF(x_pos, rect.bottom()),
                QtCore.QPointF(x_pos, rect.bottom() + _TICK_LEN),
            )
            text_width = fm.horizontalAdvance(label)
            painter.drawText(
                int(x_pos - text_width / 2),
                rect.bottom() + _TICK_LEN + fm.ascent() + 2,
                label,
            )

        for value, label in zip(layout.y_ticks, layout.y_labels, strict=False):
            y_pos = rect.bottom() - (value - y_min) / y_span * rect.height()
            painter.drawLine(
                QtCore.QPointF(rect.left() - _TICK_LEN, y_pos),
                QtCore.QPointF(rect.left(), y_pos),
            )
            text_width = fm.horizontalAdvance(label)
            painter.drawText(
                rect.left() - _TICK_LEN - text_width - 6,
                int(y_pos + fm.ascent() / 2),
                label,
            )

    def _draw_titles(
        self,
        painter: QtGui.QPainter,
        layout: _PlotLayout,
        fm: QtGui.QFontMetrics,
        axis_color: QtGui.QColor,
    ) -> None:
        rect = layout.draw_rect
        painter.setPen(axis_color)
        if self._title:
            top_margin = max(layout.margins[2], fm.height())
            title_rect = QtCore.QRect(rect.left(), 0, rect.width(), top_margin)
            painter.drawText(
                title_rect,
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                self._title,
            )
        if self._xlabel:
            bottom_margin = max(layout.margins[3], fm.height())
            xlabel_rect = QtCore.QRect(
                rect.left(),
                self.height() - bottom_margin,
                rect.width(),
                bottom_margin,
            )
            painter.drawText(
                xlabel_rect,
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                self._xlabel,
            )
        if self._ylabel:
            painter.save()
            painter.translate(fm.height(), rect.center().y())
            painter.rotate(-90)
            painter.drawText(
                -rect.height() / 2,
                -fm.descent() - 2,
                rect.height(),
                fm.height(),
                QtCore.Qt.AlignHCenter,
                self._ylabel,
            )
            painter.restore()

    def _draw_legend(
        self,
        painter: QtGui.QPainter,
        layout: _PlotLayout,
        fm: QtGui.QFontMetrics,
    ) -> None:
        entries = [series for series in self._series if series.label]
        if not entries:
            return

        padding = 8
        swatch_w = 26
        swatch_gap = 8
        text_heights = fm.height()
        max_label_w = max(fm.horizontalAdvance(series.label) for series in entries)
        row_h = max(text_heights, 12)
        width = padding * 2 + swatch_w + swatch_gap + max_label_w
        height = padding * 2 + row_h * len(entries)

        rect = QtCore.QRect(
            layout.draw_rect.right() - width - 8,
            layout.draw_rect.top() + 8,
            width,
            height,
        )
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        panel_bg, panel_border, panel_text = self._panel_colors()
        painter.setPen(QtGui.QPen(panel_border))
        painter.setBrush(panel_bg)
        painter.drawRoundedRect(rect, 6, 6)

        y_cursor = rect.top() + padding
        for series in entries:
            y_mid = y_cursor + row_h / 2
            x_start = rect.left() + padding
            x_end = x_start + swatch_w
            pen = QtGui.QPen(series.color, max(1.0, series.linewidth))
            pen.setStyle(_linestyle_to_qt(series.linestyle))
            painter.setPen(pen)
            painter.drawLine(
                QtCore.QPointF(x_start, y_mid), QtCore.QPointF(x_end, y_mid)
            )
            if series.marker:
                painter.setBrush(series.color)
                painter.setPen(QtGui.QPen(series.color, max(1.0, series.linewidth)))
                _draw_marker(
                    painter,
                    QtCore.QPointF((x_start + x_end) / 2, y_mid),
                    series.marker,
                    min(series.markersize, 10),
                )
            painter.setPen(QtGui.QPen(panel_text))
            painter.drawText(
                x_end + swatch_gap,
                int(y_mid + fm.ascent() / 2),
                series.label,
            )
            y_cursor += row_h
        painter.restore()

    def _draw_markers(
        self, painter: QtGui.QPainter, layout: _PlotLayout, fm: QtGui.QFontMetrics
    ) -> None:
        if not self._markers:
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        for marker in self._markers:
            box_info = self._marker_box_rect(marker, layout, fm)
            if box_info is None:
                continue
            box_rect, marker_pos, lines, color = box_info
            self._draw_marker_handle(painter, marker_pos, color)
            self._draw_marker_connector(painter, marker_pos, box_rect, color)
            self._draw_marker_tooltip(painter, box_rect, lines, fm)
        painter.restore()

    def _draw_marker_handle(
        self, painter: QtGui.QPainter, pos: QtCore.QPointF, color: QtGui.QColor
    ) -> None:
        outer_pen = QtGui.QPen(QtGui.QColor("white"))
        outer_pen.setWidth(_DATA_MARKER_OUTER_WIDTH)
        painter.setPen(outer_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(pos, _DATA_MARKER_RADIUS_OUTER, _DATA_MARKER_RADIUS_OUTER)

        inner_pen = QtGui.QPen(color)
        inner_pen.setWidth(_DATA_MARKER_INNER_WIDTH)
        painter.setPen(inner_pen)
        painter.drawEllipse(pos, _DATA_MARKER_RADIUS_INNER, _DATA_MARKER_RADIUS_INNER)
        painter.drawLine(
            pos + QtCore.QPointF(-_DATA_MARKER_CROSS_HALF, 0),
            pos + QtCore.QPointF(_DATA_MARKER_CROSS_HALF, 0),
        )
        painter.drawLine(
            pos + QtCore.QPointF(0, -_DATA_MARKER_CROSS_HALF),
            pos + QtCore.QPointF(0, _DATA_MARKER_CROSS_HALF),
        )

    def _draw_marker_tooltip(
        self,
        painter: QtGui.QPainter,
        box_rect: QtCore.QRectF,
        lines: list[str],
        fm: QtGui.QFontMetrics,
    ) -> None:
        panel_bg, panel_border, panel_text = self._panel_colors()
        painter.setPen(QtGui.QPen(panel_border, 1))
        painter.setBrush(panel_bg)
        painter.drawRoundedRect(
            box_rect,
            _DATA_MARKER_TOOLTIP_CORNER_RADIUS,
            _DATA_MARKER_TOOLTIP_CORNER_RADIUS,
        )

        text_y = box_rect.top() + _DATA_MARKER_TOOLTIP_V_PADDING + fm.ascent()
        painter.setPen(QtGui.QPen(panel_text))
        for line in lines:
            painter.drawText(
                box_rect.left() + _DATA_MARKER_TOOLTIP_H_PADDING, text_y, line
            )
            text_y += fm.height()

    def _draw_marker_connector(
        self,
        painter: QtGui.QPainter,
        marker_pos: QtCore.QPointF,
        box_rect: QtCore.QRectF,
        color: QtGui.QColor,
    ) -> None:
        anchor_x = (
            box_rect.left() + _DATA_MARKER_ANCHOR_OFFSET
            if marker_pos.x() <= box_rect.center().x()
            else box_rect.right() - _DATA_MARKER_ANCHOR_OFFSET
        )
        anchor_point = QtCore.QPointF(anchor_x, box_rect.center().y())
        painter.setPen(QtGui.QPen(color, 1.4))
        painter.drawLine(marker_pos, anchor_point)

    def _marker_info(
        self, marker: _PlotMarker, layout: _PlotLayout
    ) -> tuple[QtCore.QPointF, float, float, str, QtGui.QColor] | None:
        if not (0 <= marker.series_idx < len(self._series)):
            return None
        series = self._series[marker.series_idx]
        if not (0 <= marker.point_idx < series.x.size):
            return None
        x_val = float(series.x[marker.point_idx])
        y_val = float(series.y[marker.point_idx])
        if not (isfinite(x_val) and isfinite(y_val)):
            return None

        rect = layout.draw_rect
        if rect.isEmpty():
            return None
        x_min, x_max = self._view_xlim
        y_min, y_max = self._view_ylim
        x_span = max(x_max - x_min, 1e-12)
        y_span = max(y_max - y_min, 1e-12)
        x_pos = rect.left() + (x_val - x_min) / x_span * rect.width()
        y_pos = rect.bottom() - (y_val - y_min) / y_span * rect.height()
        point = QtCore.QPointF(x_pos, y_pos)
        if not rect.contains(int(x_pos), int(y_pos)):
            return None
        return point, x_val, y_val, series.label, series.color

    def _marker_box_rect(
        self,
        marker: _PlotMarker,
        layout: _PlotLayout,
        fm: QtGui.QFontMetrics,
    ) -> tuple[QtCore.QRectF, QtCore.QPointF, list[str], QtGui.QColor] | None:
        info = self._marker_info(marker, layout)
        if info is None:
            return None
        marker_pos, x_val, y_val, label, color = info
        lines = []
        if label:
            lines.append(label)
        lines.append(f"x = {_format_tick(x_val)}")
        lines.append(f"y = {_format_tick(y_val)}")

        text_width = max(fm.horizontalAdvance(line) for line in lines)
        box_width = text_width + 2 * _DATA_MARKER_TOOLTIP_H_PADDING
        box_height = len(lines) * fm.height() + 2 * _DATA_MARKER_TOOLTIP_V_PADDING

        if marker.box_offset is None:
            box_x = marker_pos.x() + _DATA_MARKER_BOX_OFFSET
            box_y = marker_pos.y() - _DATA_MARKER_BOX_OFFSET - box_height
        else:
            box_x = marker_pos.x() + marker.box_offset.x()
            box_y = marker_pos.y() + marker.box_offset.y()

        box_x = min(
            max(box_x, _DATA_MARKER_TOOLTIP_CLAMP_MARGIN),
            self.width() - _DATA_MARKER_TOOLTIP_CLAMP_MARGIN - box_width,
        )
        box_y = min(
            max(box_y, _DATA_MARKER_TOOLTIP_CLAMP_MARGIN),
            self.height() - _DATA_MARKER_TOOLTIP_CLAMP_MARGIN - box_height,
        )

        marker.box_offset = QtCore.QPointF(
            box_x - marker_pos.x(),
            box_y - marker_pos.y(),
        )
        box_rect = QtCore.QRectF(box_x, box_y, box_width, box_height)
        return box_rect, marker_pos, lines, color

    def _marker_box_hit(
        self,
        pos: QtCore.QPoint,
        layout: _PlotLayout,
        fm: QtGui.QFontMetrics,
    ) -> int | None:
        for idx, marker in enumerate(self._markers):
            box_info = self._marker_box_rect(marker, layout, fm)
            if box_info is None:
                continue
            box_rect, _, _, _ = box_info
            if box_rect.contains(QtCore.QPointF(pos)):
                return idx
        return None

    def _add_marker(self, pos: QtCore.QPoint, layout: _PlotLayout) -> None:
        nearest = self._nearest_point(pos, layout)
        if nearest is None:
            return
        series_idx, point_idx = nearest
        self._markers.append(_PlotMarker(series_idx=series_idx, point_idx=point_idx))
        self.update()

    def _nearest_point(
        self, pos: QtCore.QPoint, layout: _PlotLayout
    ) -> tuple[int, int] | None:
        rect = layout.draw_rect
        if rect.isEmpty():
            return None
        x_min, x_max = self._view_xlim
        y_min, y_max = self._view_ylim
        x_span = max(x_max - x_min, 1e-12)
        y_span = max(y_max - y_min, 1e-12)

        best: tuple[int, int, float] | None = None
        for idx, series in enumerate(self._series):
            mask = isfinite(series.x) & isfinite(series.y)
            if not mask.any():
                continue
            indices = np.flatnonzero(mask)
            x_vals = series.x[indices]
            y_vals = series.y[indices]
            x_pos = rect.left() + (x_vals - x_min) / x_span * rect.width()
            y_pos = rect.bottom() - (y_vals - y_min) / y_span * rect.height()
            dx = x_pos - pos.x()
            dy = y_pos - pos.y()
            dist2 = dx * dx + dy * dy
            local_idx = int(dist2.argmin())
            local_dist = float(dist2[local_idx])
            if best is None or local_dist < best[2]:
                best = (idx, int(indices[local_idx]), local_dist)

        if best is None:
            return None
        return best[0], best[1]

    def _show_context_menu(self, pos: QtCore.QPoint) -> None:
        def _menu_text(label: str, shortcut: str | None = None) -> str:
            if not shortcut:
                return label
            sequence = QtGui.QKeySequence(shortcut)
            return f"{label} ({sequence.toString(QtGui.QKeySequence.NativeText)})"

        layout = self._last_layout or self._layout(self.fontMetrics())
        menu = QtWidgets.QMenu(self)
        zoom_action = menu.addAction(_menu_text("Zoom", "Z"))
        zoom_x_action = menu.addAction(_menu_text("Zoom X", "H"))
        zoom_y_action = menu.addAction(_menu_text("Zoom Y", "V"))
        pan_action = menu.addAction("Pan mode")
        mode_group = QtGui.QActionGroup(menu)
        for act in (zoom_action, zoom_x_action, zoom_y_action, pan_action):
            act.setCheckable(True)
            mode_group.addAction(act)
        mode_group.setExclusive(True)

        menu.addSeparator()
        reset_action = menu.addAction(_menu_text("Reset view", "R"))

        menu.addSeparator()
        grid_action = menu.addAction("Grid")
        grid_action.setCheckable(True)
        grid_action.setChecked(self._grid)
        legend_action = menu.addAction("Legend")
        legend_action.setCheckable(True)
        legend_action.setChecked(self._legend)

        menu.addSeparator()
        add_marker_action = menu.addAction("Add data marker here")
        clear_markers_action = menu.addAction("Clear data markers")

        menu.addSeparator()
        copy_figure_action = menu.addAction(_menu_text("Copy figure to clipboard", "Y"))
        copy_axes_action = menu.addAction(_menu_text("Copy axes to clipboard", "Shift+Y"))
        save_png_action = menu.addAction(_menu_text("Save figure as PNG...", "S"))

        if self._zoom_axis_mode == "xy":
            zoom_action.setChecked(True)
        elif self._zoom_axis_mode == "x":
            zoom_x_action.setChecked(True)
        elif self._zoom_axis_mode == "y":
            zoom_y_action.setChecked(True)
        else:
            pan_action.setChecked(True)

        chosen = menu.exec(self.mapToGlobal(pos))
        if chosen == zoom_action:
            self.set_zoom_axis_mode("xy")
        elif chosen == zoom_x_action:
            self.set_zoom_axis_mode("x")
        elif chosen == zoom_y_action:
            self.set_zoom_axis_mode("y")
        elif chosen == pan_action:
            self.set_zoom_axis_mode("off")
        elif chosen == reset_action:
            self.reset_view()
        elif chosen == grid_action:
            self.set_grid(grid_action.isChecked())
        elif chosen == legend_action:
            self.set_legend(legend_action.isChecked())
        elif chosen == add_marker_action:
            if layout.draw_rect.contains(pos):
                self._add_marker(pos, layout)
        elif chosen == clear_markers_action:
            self._markers.clear()
            self.update()
        elif chosen == copy_figure_action:
            self.copy_figure_to_clipboard()
        elif chosen == copy_axes_action:
            self.copy_axes_to_clipboard()
        elif chosen == save_png_action:
            self.save_figure_to_png()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        layout = self._last_layout
        if layout is None or layout.draw_rect.isEmpty():
            return
        if not layout.draw_rect.contains(event.position().toPoint()):
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.1 if delta > 0 else 1 / 1.1
        data_x, data_y = _screen_to_data(
            event.position(), layout.draw_rect, self._view_xlim, self._view_ylim
        )
        self._apply_zoom(data_x, data_y, factor, self._zoom_axis_mode)
        self.update()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != QtCore.Qt.LeftButton:
            return
        fm = QtGui.QFontMetrics(self.font())
        layout = self._last_layout or self._layout(fm)
        if not layout.draw_rect.contains(event.position().toPoint()):
            return
        hit_box = self._marker_box_hit(event.position().toPoint(), layout, fm)
        if hit_box is not None:
            self._drag_target = {"kind": "box", "idx": hit_box}
            self._box_drag_start = event.position().toPoint()
            event.accept()
            return
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            self._add_marker(event.position().toPoint(), layout)
            event.accept()
            return
        self._drag_target = None
        self._dragging = True
        self._drag_start = event.position().toPoint()
        self._pan_xlim_start = self._view_xlim
        self._pan_ylim_start = self._view_ylim
        if self._zoom_axis_mode != "off":
            self._rubber_band = QtCore.QRect(self._drag_start, self._drag_start)
        else:
            self._rubber_band = None

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._drag_target is not None:
            kind = self._drag_target.get("kind")
            idx = int(self._drag_target.get("idx", -1))
            if kind == "box" and 0 <= idx < len(self._markers):
                marker = self._markers[idx]
                delta = event.position().toPoint() - self._box_drag_start
                current_offset = marker.box_offset or QtCore.QPointF(0, 0)
                marker.box_offset = QtCore.QPointF(
                    current_offset.x() + delta.x(),
                    current_offset.y() + delta.y(),
                )
                self._box_drag_start = event.position().toPoint()
                self.update()
                event.accept()
                return
        if not self._dragging:
            return
        layout = self._last_layout
        if layout is None:
            return
        pos = event.position().toPoint()
        if self._zoom_axis_mode != "off":
            self._rubber_band = _selection_rect(
                layout.draw_rect, self._drag_start, pos, self._zoom_axis_mode
            )
        else:
            self._apply_pan(pos, layout.draw_rect)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._drag_target is not None:
            self._drag_target = None
            event.accept()
            return
        if not self._dragging:
            return
        self._dragging = False
        if self._zoom_axis_mode != "off" and self._rubber_band is not None:
            self._apply_box_zoom(self._rubber_band, self._zoom_axis_mode)
        self._rubber_band = None
        self.update()

    def _apply_pan(self, pos: QtCore.QPoint, rect: QtCore.QRect) -> None:
        dx = pos.x() - self._drag_start.x()
        dy = pos.y() - self._drag_start.y()
        x_min, x_max = self._pan_xlim_start
        y_min, y_max = self._pan_ylim_start
        if rect.width() == 0 or rect.height() == 0:
            return
        x_shift = -dx / rect.width() * (x_max - x_min)
        y_shift = dy / rect.height() * (y_max - y_min)
        self._view_xlim = (x_min + x_shift, x_max + x_shift)
        self._view_ylim = (y_min + y_shift, y_max + y_shift)

    def _apply_zoom(
        self, data_x: float, data_y: float, factor: float, mode: str
    ) -> None:
        x_min, x_max = self._view_xlim
        y_min, y_max = self._view_ylim
        if mode in {"xy", "x", "off"}:
            x_range = max(x_max - x_min, 1e-12)
            new_range = x_range / factor
            rel = (data_x - x_min) / x_range
            new_x_min = data_x - rel * new_range
            self._view_xlim = (new_x_min, new_x_min + new_range)
        if mode in {"xy", "y", "off"}:
            y_range = max(y_max - y_min, 1e-12)
            new_range = y_range / factor
            rel = (data_y - y_min) / y_range
            new_y_min = data_y - rel * new_range
            self._view_ylim = (new_y_min, new_y_min + new_range)

    def _apply_box_zoom(self, rect: QtCore.QRect, mode: str) -> None:
        if rect.width() < 4 and rect.height() < 4:
            return
        layout = self._last_layout or self._layout(self.fontMetrics())
        draw_rect = layout.draw_rect
        x0, y0 = _screen_to_data(
            rect.topLeft(), draw_rect, self._view_xlim, self._view_ylim
        )
        x1, y1 = _screen_to_data(
            rect.bottomRight(), draw_rect, self._view_xlim, self._view_ylim
        )
        x_min, x_max = sorted((x0, x1))
        y_min, y_max = sorted((y0, y1))

        cur_x_min, cur_x_max = self._view_xlim
        cur_y_min, cur_y_max = self._view_ylim

        if mode in {"xy", "x"}:
            self._view_xlim = (x_min, x_max) if x_min != x_max else (cur_x_min, cur_x_max)
        if mode in {"xy", "y"}:
            self._view_ylim = (y_min, y_max) if y_min != y_max else (cur_y_min, cur_y_max)


class PlotWindow(QtWidgets.QMainWindow):
    """Basic window that displays 1D plots with grid, legend, and toolbar."""

    def __init__(
        self,
        *,
        title: str = "plotqt",
        xlabel: str = "",
        ylabel: str = "",
        grid: bool = False,
        legend: bool = False,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self._canvas = _PlotCanvas(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            grid=grid,
            legend=legend,
            xlim=xlim,
            ylim=ylim,
        )
        self.setCentralWidget(self._canvas)
        self._toolbar = FigureToolbar(self._canvas, parent=self)
        self._toolbar.setObjectName("qtplotlibToolbar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self._toolbar)
        self.resize(800, 560)

    def plot(
        self,
        *args: object,
        label: str | Sequence[str] | None = None,
        color: str | QtGui.QColor | Sequence[str | QtGui.QColor] | None = None,
        linestyle: str | Sequence[str] | None = None,
        marker: str | Sequence[str] | None = None,
        linewidth: float | Sequence[float] = 1.6,
        markersize: float | Sequence[float] = 6.0,
        alpha: float | Sequence[float] = 1.0,
    ) -> None:
        """Add one or more series using plot-style arguments.

        Args:
            *args: Plot-style data arguments, e.g. (y), (x, y), or (x, y, fmt).
            label: Label(s) for legend entries.
            color: Color(s) for the series.
            linestyle: Line style(s) for the series.
            marker: Marker style(s) for the series.
            linewidth: Line width(s).
            markersize: Marker size(s).
            alpha: Opacity value(s) between 0 and 1.
        """
        series_defs = _parse_plot_args(args)
        labels = _coerce_sequence(label, len(series_defs))
        colors = _coerce_sequence(color, len(series_defs))
        linestyles = _coerce_sequence(linestyle, len(series_defs))
        markers = _coerce_sequence(marker, len(series_defs))
        linewidths = _coerce_sequence(linewidth, len(series_defs))
        markersizes = _coerce_sequence(markersize, len(series_defs))
        alphas = _coerce_sequence(alpha, len(series_defs))

        series_list: list[_PlotSeries] = []
        for (x_val, y_val, fmt), lbl, col, ls, mk, lw, ms, al in zip(
            series_defs,
            labels,
            colors,
            linestyles,
            markers,
            linewidths,
            markersizes,
            alphas,
            strict=False,
        ):
            series_list.append(
                _build_series(
                    x_val,
                    y_val,
                    fmt=fmt,
                    label=str(lbl) if lbl is not None else "",
                    color=col,
                    linestyle=ls,
                    marker=mk,
                    linewidth=lw,
                    markersize=ms,
                    alpha=al,
                    canvas=self._canvas,
                )
            )
        self._canvas.add_series(series_list)

    def set_plot(
        self,
        *args: object,
        label: str | Sequence[str] | None = None,
        color: str | QtGui.QColor | Sequence[str | QtGui.QColor] | None = None,
        linestyle: str | Sequence[str] | None = None,
        marker: str | Sequence[str] | None = None,
        linewidth: float | Sequence[float] = 1.6,
        markersize: float | Sequence[float] = 6.0,
        alpha: float | Sequence[float] = 1.0,
    ) -> None:
        """Replace existing series using plot-style arguments."""
        series_defs = _parse_plot_args(args)
        labels = _coerce_sequence(label, len(series_defs))
        colors = _coerce_sequence(color, len(series_defs))
        linestyles = _coerce_sequence(linestyle, len(series_defs))
        markers = _coerce_sequence(marker, len(series_defs))
        linewidths = _coerce_sequence(linewidth, len(series_defs))
        markersizes = _coerce_sequence(markersize, len(series_defs))
        alphas = _coerce_sequence(alpha, len(series_defs))

        series_list: list[_PlotSeries] = []
        for (x_val, y_val, fmt), lbl, col, ls, mk, lw, ms, al in zip(
            series_defs,
            labels,
            colors,
            linestyles,
            markers,
            linewidths,
            markersizes,
            alphas,
            strict=False,
        ):
            series_list.append(
                _build_series(
                    x_val,
                    y_val,
                    fmt=fmt,
                    label=str(lbl) if lbl is not None else "",
                    color=col,
                    linestyle=ls,
                    marker=mk,
                    linewidth=lw,
                    markersize=ms,
                    alpha=al,
                    canvas=self._canvas,
                )
            )
        self._canvas.set_series(series_list)

    def clear(self) -> None:
        """Clear all plotted series."""
        self._canvas.clear()

    def set_xlabel(self, text: str) -> None:
        self._canvas.set_xlabel(text)

    def set_ylabel(self, text: str) -> None:
        self._canvas.set_ylabel(text)

    def set_title(self, text: str) -> None:
        self._canvas.set_title(text)
        self.setWindowTitle(text)

    def set_grid(self, enabled: bool) -> None:
        self._canvas.set_grid(enabled)

    def set_legend(self, enabled: bool) -> None:
        self._canvas.set_legend(enabled)

    def set_xlim(self, xlim: tuple[float, float]) -> None:
        self._canvas.set_xlim(xlim)

    def set_ylim(self, ylim: tuple[float, float]) -> None:
        self._canvas.set_ylim(ylim)

    def tight_layout(self, *, pad: float = 1.08) -> None:
        """Enable compact margins around the plot."""
        self._canvas.tight_layout(pad=pad)

    def disable_tight_layout(self) -> None:
        """Disable compact margins around the plot."""
        self._canvas.disable_tight_layout()


def plotqt(
    *args: object,
    title: str = "plotqt",
    xlabel: str = "",
    ylabel: str = "",
    grid: bool = False,
    legend: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    label: str | Sequence[str] | None = None,
    color: str | QtGui.QColor | Sequence[str | QtGui.QColor] | None = None,
    linestyle: str | Sequence[str] | None = None,
    marker: str | Sequence[str] | None = None,
    linewidth: float | Sequence[float] = 1.6,
    markersize: float | Sequence[float] = 6.0,
    alpha: float | Sequence[float] = 1.0,
) -> PlotWindow:
    """Plot 1D data in a PySide6 window with a Matplotlib-like toolbar.

    Args:
        *args: Plot-style data arguments, e.g. (y), (x, y), or (x, y, fmt).
        title: Window title.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        grid: Whether to show grid lines.
        legend: Whether to show the legend.
        xlim: Optional x-axis limits.
        ylim: Optional y-axis limits.
        label: Label(s) for legend entries.
        color: Color(s) for the series.
        linestyle: Line style(s) for the series.
        marker: Marker style(s) for the series.
        linewidth: Line width(s).
        markersize: Marker size(s).
        alpha: Opacity value(s) between 0 and 1.
    """
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        app = QtWidgets.QApplication([])
        created_app = True

    window = PlotWindow(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
        legend=legend,
        xlim=xlim,
        ylim=ylim,
    )
    if args:
        window.set_plot(
            *args,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=linewidth,
            markersize=markersize,
            alpha=alpha,
        )
    window.show()

    if created_app:
        app.exec()
    return window


def _build_series(
    x_val: object | None,
    y_val: object,
    *,
    fmt: str | None,
    label: str,
    color: object | None,
    linestyle: object | None,
    marker: object | None,
    linewidth: object | None,
    markersize: object | None,
    alpha: object | None,
    canvas: _PlotCanvas,
) -> _PlotSeries:
    y_arr = _coerce_1d(y_val, "y")
    if x_val is None:
        x_arr = np.arange(y_arr.size, dtype=float)
    else:
        x_arr = _coerce_1d(x_val, "x")
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length.")

    fmt_style = _parse_format_string(fmt)
    resolved_color = color if color is not None else fmt_style.color
    if resolved_color is None:
        resolved_color = _DEFAULT_COLORS[canvas._color_index % len(_DEFAULT_COLORS)]
    qcolor = _to_qcolor(resolved_color)
    canvas._color_index += 1

    resolved_linestyle = (
        str(linestyle) if linestyle is not None else fmt_style.linestyle
    )
    if resolved_linestyle is None:
        resolved_linestyle = "-"

    resolved_marker = str(marker) if marker is not None else fmt_style.marker

    resolved_linewidth = float(linewidth) if linewidth is not None else 1.6
    resolved_markersize = float(markersize) if markersize is not None else 6.0
    resolved_alpha = float(alpha) if alpha is not None else 1.0

    return _PlotSeries(
        x=x_arr,
        y=y_arr,
        label=label,
        color=qcolor,
        linestyle=resolved_linestyle,
        marker=resolved_marker,
        linewidth=resolved_linewidth,
        markersize=resolved_markersize,
        alpha=resolved_alpha,
    )


def _pad_limits(min_val: float, max_val: float) -> tuple[float, float]:
    if min_val == max_val:
        pad = 1.0 if min_val == 0 else abs(min_val) * 0.05
        return min_val - pad, max_val + pad
    span = max_val - min_val
    pad = span * 0.05
    return min_val - pad, max_val + pad


def _linestyle_to_qt(linestyle: str) -> QtCore.Qt.PenStyle:
    if linestyle in {"none", "None", "", " "}:
        return QtCore.Qt.NoPen
    if linestyle == "--":
        return QtCore.Qt.DashLine
    if linestyle == "-.":
        return QtCore.Qt.DashDotLine
    if linestyle == ":":
        return QtCore.Qt.DotLine
    return QtCore.Qt.SolidLine


def _flush_line(painter: QtGui.QPainter, points: list[QtCore.QPointF]) -> None:
    if len(points) < 2:
        return
    painter.drawPolyline(QtGui.QPolygonF(points))


def _draw_marker(
    painter: QtGui.QPainter,
    pos: QtCore.QPointF,
    marker: str,
    size: float,
) -> None:
    half = size / 2
    if marker == "o":
        painter.drawEllipse(pos, half, half)
    elif marker == "s":
        painter.drawRect(QtCore.QRectF(pos.x() - half, pos.y() - half, size, size))
    elif marker == "^":
        path = QtGui.QPainterPath()
        path.moveTo(pos.x(), pos.y() - half)
        path.lineTo(pos.x() + half, pos.y() + half)
        path.lineTo(pos.x() - half, pos.y() + half)
        path.closeSubpath()
        painter.drawPath(path)
    elif marker == "v":
        path = QtGui.QPainterPath()
        path.moveTo(pos.x() - half, pos.y() - half)
        path.lineTo(pos.x() + half, pos.y() - half)
        path.lineTo(pos.x(), pos.y() + half)
        path.closeSubpath()
        painter.drawPath(path)
    elif marker == "D":
        path = QtGui.QPainterPath()
        path.moveTo(pos.x(), pos.y() - half)
        path.lineTo(pos.x() + half, pos.y())
        path.lineTo(pos.x(), pos.y() + half)
        path.lineTo(pos.x() - half, pos.y())
        path.closeSubpath()
        painter.drawPath(path)
    elif marker == "x":
        painter.drawLine(
            QtCore.QPointF(pos.x() - half, pos.y() - half),
            QtCore.QPointF(pos.x() + half, pos.y() + half),
        )
        painter.drawLine(
            QtCore.QPointF(pos.x() - half, pos.y() + half),
            QtCore.QPointF(pos.x() + half, pos.y() - half),
        )
    elif marker == "+":
        painter.drawLine(
            QtCore.QPointF(pos.x() - half, pos.y()),
            QtCore.QPointF(pos.x() + half, pos.y()),
        )
        painter.drawLine(
            QtCore.QPointF(pos.x(), pos.y() - half),
            QtCore.QPointF(pos.x(), pos.y() + half),
        )
    elif marker == "*":
        painter.drawLine(
            QtCore.QPointF(pos.x() - half, pos.y()),
            QtCore.QPointF(pos.x() + half, pos.y()),
        )
        painter.drawLine(
            QtCore.QPointF(pos.x(), pos.y() - half),
            QtCore.QPointF(pos.x(), pos.y() + half),
        )
        painter.drawLine(
            QtCore.QPointF(pos.x() - half, pos.y() - half),
            QtCore.QPointF(pos.x() + half, pos.y() + half),
        )
        painter.drawLine(
            QtCore.QPointF(pos.x() - half, pos.y() + half),
            QtCore.QPointF(pos.x() + half, pos.y() - half),
        )
    elif marker == ".":
        painter.drawEllipse(pos, max(1.0, half * 0.4), max(1.0, half * 0.4))


def _selection_rect(
    draw_rect: QtCore.QRect,
    start: QtCore.QPoint,
    current: QtCore.QPoint,
    mode: str,
) -> QtCore.QRect:
    start_x = max(min(start.x(), draw_rect.right()), draw_rect.left())
    start_y = max(min(start.y(), draw_rect.bottom()), draw_rect.top())
    cur_x = max(min(current.x(), draw_rect.right()), draw_rect.left())
    cur_y = max(min(current.y(), draw_rect.bottom()), draw_rect.top())

    if mode == "x":
        return QtCore.QRect(
            QtCore.QPoint(min(start_x, cur_x), draw_rect.top()),
            QtCore.QPoint(max(start_x, cur_x), draw_rect.bottom()),
        ).normalized()
    if mode == "y":
        return QtCore.QRect(
            QtCore.QPoint(draw_rect.left(), min(start_y, cur_y)),
            QtCore.QPoint(draw_rect.right(), max(start_y, cur_y)),
        ).normalized()

    rect = QtCore.QRect(
        QtCore.QPoint(start_x, start_y), QtCore.QPoint(cur_x, cur_y)
    ).normalized()
    return rect & draw_rect


def _screen_to_data(
    pos: QtCore.QPointF | QtCore.QPoint,
    rect: QtCore.QRect,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[float, float]:
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    x_val = x_min + (pos.x() - rect.left()) / rect.width() * x_span
    y_val = y_max - (pos.y() - rect.top()) / rect.height() * y_span
    return x_val, y_val
