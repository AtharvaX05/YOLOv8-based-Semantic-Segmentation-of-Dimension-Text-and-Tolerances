# 📐 Technical Drawing OCR Pipeline

A lightweight yet powerful pipeline for detecting and extracting dimension text from technical drawings. Combines YOLO OBB (Oriented Bounding Box) detection with Tesseract OCR to convert drawing images into structured dimension data.

---

## 🎯 Project Overview

This system automatically:
- **Detects** dimension regions in technical drawings using YOLOv8 OBB
- **Localizes** rotated text callouts with high precision
- **Preprocesses** image crops for optimal OCR readability
- **Extracts** dimension values using Tesseract OCR
- **Exports** annotated images and text reports

**Perfect for:** Engineering drawing digitization, dimension automation, blueprint analysis, and CAD data extraction.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📐 **OBB Detection** | Oriented bounding box model for rotated dimension text |
| 🎯 **Class Targeting** | Filters specifically for dimension callouts (class ID: 1) |
| 🔄 **Rotation Correction** | Automatically normalizes crop orientation before OCR |
| 🖼️ **Smart Preprocessing** | Grayscale conversion, upscaling, and Otsu thresholding |
| ⚙️ **Optimized OCR Config** | Tuned character whitelist and Tesseract parameters for technical text |
| 📤 **Dual Output** | Saves both annotated visuals and structured text reports |
| 🪟 **Windows Ready** | Built-in support for Windows Tesseract installation paths |

---

## 💻 Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8 OBB (Ultralytics) |
| Deep Learning | PyTorch + Torchvision |
| OCR Engine | Tesseract (Open Source) |
| Image Processing | OpenCV, NumPy |
| Language | Python 3.10+ |

---

## 📋 Requirements

- 🐍 **Python 3.10+**
- 🔤 **Tesseract OCR** (installed locally)
- 📦 **Python dependencies** from `requirements.txt`
- 🖥️ **RAM:** 4GB minimum (8GB recommended for full datasets)

---

## 🛠️ Installation & Setup

### 1️⃣ Clone & Setup Python Environment

```bash
# Navigate to project directory
cd your-project-folder

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Install Tesseract OCR

**Windows:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (default path: `C:\Program Files\Tesseract-OCR`)
3. The script auto-detects this path

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 4️⃣ Configure Paths (if needed)

Edit `testocr.py` if Tesseract is installed elsewhere:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Update model and image paths:
```python
model_path = "runs/obb/technical_drawing/weights/best.pt"
image_path = "path/to/your/drawing.jpg"
```

---

## ▶️ Usage

Run the pipeline on a technical drawing:

```bash
python testocr.py
```

**What happens:**
1. Detects dimension regions in the input image
2. Normalizes rotations and preprocesses crops
3. Runs OCR to extract dimension text
4. Exports results to `output5/`

---

## 📁 Output Format

### Files Generated:
- **`output5/annotated_result.jpg`** — Original image with detected dimensions highlighted and labeled
- **`output5/results.txt`** — Structured text report with extracted dimension values

### Example `results.txt`:
```
TECHNICAL DRAWING OCR REPORT
Generated: 2026-03-24 10:30:45

1. 25.5
2. 12.0
3. ∅8.5
4. +0.2/-0.1
5. R10
...
```

---

## 🧪 Sample Annotated Output

### Example Detection Result:

![Sample Annotated Output](output/annotated_result.jpg)

<br>
<br>

**Additional Examples & Variations:**

*This space is reserved for more annotated samples, edge cases, and comparison panels. Add multiple examples to demonstrate pipeline robustness across different drawing styles.*

<br>
<br>

---

## 🔧 Configuration & Tuning

### Detection Confidence
```python
results = model.predict(source=image_path, conf=0.25)  # Adjust 0.25 threshold
```

### OCR Settings
```python
TESS_CONFIG = (
    "--oem 3 "           # OCR Engine Mode
    "--psm 7 "           # Page Segmentation Mode
    "-c tessedit_char_whitelist=0123456789MRrØ°.-"  # Allowed characters
)
```

### Preprocessing
```python
# Grayscale conversion, upscaling, thresholding
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

---

## 📊 Project Structure

```
├── requirements.txt                 # Python dependencies
├── testocr.py                      # Main pipeline script
├── yolov8n-obb.pt                  # OBB model weights
├── dataset-views/                  # Training data
│   ├── images/
│   └── labels/
├── output5/                        # Results (auto-created)
│   ├── annotated_result.jpg
│   └── results.txt
└── runs/
    └── obb/                        # YOLOv8 training outputs
```

---

## 🎓 Model Details

- **Architecture:** YOLOv8 Oriented Bounding Box (OBB)
- **Input Size:** Variable (optimized for 640x640+)
- **Classes:** 3 total (focus on "dimension" class)
  - Class 0: View
  - Class 1: **Dimension** ✓ (primary target)
  - Class 2: Balloon
- **Performance:** Real-time inference on CPU, optimized for GPU

---

## 💡 Best Practices

✅ **Do:**
- Use high-resolution technical drawings (600+ DPI)
- Ensure good lighting and contrast in images
- Test with sample drawings before batch processing
- Validate OCR output against original documents
- Adjust thresholds based on drawing quality

❌ **Don't:**
- Use low-quality or heavily noisy scans
- Expect 100% OCR accuracy without preprocessing
- Process images with extreme rotations (>45°)
- Mix drawing styles without retraining

---

## 🚀 Future Enhancements

| Upgrade | Impact |
|---------|--------|
| YOLOv8-OBB model retraining | Higher accuracy on varied drawing styles |
| PaddleOCR integration | Superior accuracy on blueprint typography |
| Auto-dimension linking | Connect dimensions to CAD geometries (graph-based) |
| Batch processing API | Handle multiple drawings efficiently |
| Tolerance parsing | Dedicated module for ±/tolerance components |
| CSV/JSON export | Direct CAD-CAM workflow integration |

---

## 👤 Author & Contact

**Atharva Yeole**
- 📧 Email: [yeoleatharva2005@gmail.com](mailto:yeoleatharva2005@gmail.com)
- 🔗 GitHub: [@AtharvaX05](https://github.com/AtharvaX05)

---

## 📝 Acknowledgments

This project builds upon the foundational work in the YOLOv8-based technical drawing analysis domain. Special thanks to the Ultralytics and Tesseract communities for their excellent libraries.

---

## 📜 License

This project is provided as-is for educational and research purposes. Feel free to use, modify, and distribute with proper attribution.

---

**Last Updated:** March 2026  
**Status:** ✅ Active Development
