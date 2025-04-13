
# 🧠 Zero-Shot Object Detection with OWL-ViT

Real-time object detection using [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) (Zero-Shot Vision Transformer) with a custom Tkinter-based GUI. Supports live webcam detection, editable prompts, prompt history, logging, and frame rate optimization.

---

## 📸 Features

- ✍️ **Live editable text prompts** for zero-shot object detection
- 📚 **Prompt history + undo**
- 💾 **Save/load prompts to file**
- 🧠 **Built on OWL-ViT (Vision Transformer) from HuggingFace**
- 🖥️ **Live webcam feed with bounding boxes**
- 📈 **Logs predictions to CSV (optional)**
- ⚡ **Torch GPU acceleration**
- 🎯 **Frame-rate optimized (target ≥10 FPS)**
- 📏 **Input resizing for performance tuning**
- 📦 Optional: **ONNX/TorchScript inference**

---

## 🛠️ Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/)
- `opencv-python`
- `Pillow`
- `numpy`

Install dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers opencv-python pillow numpy
```

---

## 🚀 Run It

```bash
python main.py
```

> Make sure your webcam is accessible.

---

## 🧪 Example Prompts

- `monitor`
- `keyboard`
- `mug`
- `a translucent purple water bottle`
- `a transparent wireless computer mouse with a visible scroll wheel`
- `a thin cylindrical transparent blue pen`

---

## 📂 Logging

Predictions (bounding boxes, labels, confidence) are saved to:

```
./logs/detections.csv
```

Format:
```csv
timestamp,label,score,x1,y1,x2,y2
```

---

## ⚙️ Optional Optimizations

- Resize input frames to improve speed: `320x240`, `416x416`, etc.
- Enable TorchScript for faster inference:

```python
detector = ObjectDetector(use_torchscript=True)
```

- Use frame skipping and caching to boost FPS
- ONNX support (WIP)

---

## 🧠 Model

This project uses [OWL-ViT (Object detection with language prompts)](https://huggingface.co/google/owlvit-base-patch32).

---

## 📸 Screenshot

![screenshot](docs/screenshot.png)

---

## 📜 License

MIT License

---

## 🙌 Credits

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [OWL-ViT Paper](https://arxiv.org/abs/2205.06230)

