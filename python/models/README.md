## YOLOv8 Segmentation Model

This app expects a **CPU-only YOLOv8 segmentation model** for people.

For the current Python-based MVP, download the Ultralytics `yolov8n-seg.pt` weights
and place them in this directory:

- `models/yolov8n-seg.pt`

Then run:

```bash
pip install ultralytics opencv-python numpy
```

The `python/segment.py` script will default to `yolov8n-seg.pt` in the project root,
but you can override the path with `--model_path` if needed.


