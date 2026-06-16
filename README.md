# qtplotlib

Quick PySide6 helper that mirrors MATLAB's `imagesc` for simple array viewing.

## Install (Python 3.11+)

```
pip install -r requirements.txt
```

## Usage

```python
%gui qt  # when using IPython/Jupyter
import numpy as np
from qtplotlib import imagescqt

data = np.random.rand(200, 200)
imagescqt(data)  # opens a Qt window (interpolation='nearest', cmap='viridis')

# with axes and aspect control (MATLAB imagesc style)
x = np.linspace(-5, 5, data.shape[1])
y = np.linspace(-2, 2, data.shape[0])
win = imagescqt(
    x,
    y,
    data,
    aspect="equal",           # 'equal' keeps data aspect, 'auto' fills space
    xlabel="X label",
    ylabel="Y label",
    title="My image",
    colorbar=True,
    colorbar_label="Amplitude",
)
# Later in code:
win.set_title("Updated title")
win.set_xlabel("Time (s)")
win.set_ylabel("Depth (m)")
win.set_colorbar_label("Power (dB)")
# You can add a colorbar later too:
win.add_colorbar(label="Added later")

# Colored masks can be overlaid on the grayscale/base image:
mask = data > np.percentile(data, 90)
mask_layer = win.add_mask(mask, color="tab:red", alpha=0.35, label="high values")
mask_layer.set_visible(True)
mask_layer.set_alpha(0.45)
# Replace all mask overlays with one layer, or remove them:
win.set_mask(mask, color="cyan", alpha=0.25)
win.clear_masks()

# Reduce empty margins similar to matplotlib.tight_layout:
win.tight_layout(pad=1.05, auto_resize=True)  # w_pad/h_pad/rect also supported
# Disable or keep window size fixed by turning it off or passing auto_resize=False:
win.disable_tight_layout()

# Data markers can be saved to and loaded from JSON.
# The stored dict has x, y, idx_col, idx_row, and value keys.
win.save_markers("markers.json")
win.load_markers("markers.json")

# In Spyder/IPython, this stores the same dict in the Variable Explorer as
# imagescqt_markers. The toolbar memory buttons use the same default name.
win.save_markers_to_memory()
win.load_markers_from_memory()
```

Mouse/keyboard notes:
- `Shift` + left-click drops a MATLAB-style data marker showing x/y/value (multiple markers are allowed); drag any marker to move it (snaps to nearest sample).
- Drag each marker's info box to reposition it; the leader line stays attached to that marker, MATLAB-style.
- Use the toolbar marker-tooltip button to show or hide marker info boxes while keeping marker handles visible.
- Delete a marker via `Delete`/`Backspace` (clears all markers) or right-click on a specific marker/info box and pick "Delete marker".
- Toolbar marker buttons save/load markers as JSON, or save/load them through the default `imagescqt_markers` memory variable for Spyder/IPython workflows.

The helper will attach to an existing Qt application (ideal for `%gui qt`). If no
`QApplication` is running, it creates one and blocks until the window closes.

`interpolation` options:
- `nearest` (default; MATLAB-style blocky pixels)
- `bilinear` (smooth scaling)

`cmap` options:
- Any matplotlib colormap name (default: `viridis`, e.g., `gray`, `plasma`, `jet`, `inferno`, …)

`aspect` options:
- `equal` (default; preserves data aspect based on x/y spans)
- `auto` (fills available space)

`colorbar`:
- `colorbar=True` to show colorbar; `colorbar_label` to set its label (also enables it).
- Runtime updates: `win.set_colorbar(True/False)`, `win.set_colorbar_label("Label")`.

`mask` overlays:
- Initial overlay: `imagescqt(data, cmap="gray", mask=mask, mask_color="tab:red", mask_alpha=0.35)`.
- Runtime overlays: `win.add_mask(mask, color="cyan", alpha=0.25)` returns a `MaskLayer`.
- Layer updates: `layer.set_data(mask)`, `layer.set_color("yellow")`, `layer.set_alpha(0.4)`, `layer.set_visible(False)`, `layer.remove()`.
- Convenience updates: `win.set_mask(mask, color="red", alpha=0.35)` replaces all masks; `win.clear_masks()` removes them.
