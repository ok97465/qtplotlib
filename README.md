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
# Reduce empty margins similar to matplotlib.tight_layout:
win.tight_layout(pad=1.05, auto_resize=True)  # w_pad/h_pad/rect also supported
# Disable or keep window size fixed by turning it off or passing auto_resize=False:
win.disable_tight_layout()
```

Mouse/keyboard notes:
- `Shift` + left-click drops a MATLAB-style data marker showing x/y/value (multiple markers are allowed); drag any marker to move it (snaps to nearest sample).
- Drag each marker's info box to reposition it; the leader line stays attached to that marker, MATLAB-style.
- Delete a marker via `Delete`/`Backspace` (clears all markers) or right-click on a specific marker/info box and pick "Delete marker".

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
