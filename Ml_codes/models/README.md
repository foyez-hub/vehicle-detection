# Models

This directory stores the trained YOLO weights used for inference and tracking.

## 📄 Files

- `best.pt`: The PyTorch model weights (recommended for local Python inference).
- `best.onnx`: The ONNX version of the model (optimized for cross-platform and mobile deployment).

## 🚀 Usage

You can specify which model to use in the scripts using the `--model` argument:

```bash
python run_gui.py --model "models/best.pt"
# OR
python run_gui.py --model "models/best.onnx"
```
