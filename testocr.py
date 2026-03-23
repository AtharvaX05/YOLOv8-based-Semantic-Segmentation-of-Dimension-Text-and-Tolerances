import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pytesseract
from datetime import datetime

# OPTIONAL: set explicitly if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ------------------------- CONFIG -------------------------

TESS_CONFIG = (
    "--oem 3 "
    "--psm 7 "
    "-c tessedit_char_whitelist=0123456789MRrØ°.-"
)

CLASSES = {0: "view", 1: "dimension", 2: "balloon"}


# ------------------------- CORE CLASS -------------------------

class TechnicalDrawingDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    # --------------------------------------------------------

    def rotate_crop(self, crop, points):
        pts = np.array(points, dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        angle = rect[-1]

        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        if abs(angle) < 2:
            return crop

        h, w = crop.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

        return cv2.warpAffine(
            crop, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )

    # --------------------------------------------------------

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    # --------------------------------------------------------

    def detect(self, image_path, conf=0.25):
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]

        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            device="cpu",
            verbose=False
        )[0]

        detections = []

        if not hasattr(results, "obb") or results.obb is None:
            return detections, img

        for obb in results.obb:
            cls = int(obb.cls[0])
            if cls != 1:
                continue

            pts = obb.xyxyxyxy[0].cpu().numpy()
            x1, y1 = pts.min(axis=0).astype(int)
            x2, y2 = pts.max(axis=0).astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            detections.append({
                "points": pts.tolist(),
                "bbox": [x1, y1, x2, y2],
                "crop": crop
            })

        return detections, img

    # --------------------------------------------------------

    def ocr_dimensions(self, detections):
        for d in detections:
            try:
                rot = self.rotate_crop(d["crop"], d["points"])
                prep = self.preprocess(rot)

                raw = pytesseract.image_to_string(prep, config=TESS_CONFIG)
                print("RAW OCR:", repr(raw))
                text = raw.strip()


                d["text"] = text if text else "?"
            except Exception as e:
                print("OCR EXCEPTION:", repr(e))
                d["text"] = "ERROR"


        return detections

    # --------------------------------------------------------

    def visualize(self, img, detections, out_path):
        for d in detections:
            pts = np.array(d["points"], np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            x, y = pts[0]
            cv2.putText(
                img, d["text"], (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        cv2.imwrite(str(out_path), img)

    # --------------------------------------------------------

    def save_txt(self, detections, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("TECHNICAL DRAWING OCR REPORT\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            for i, d in enumerate(detections, 1):
                f.write(f"{i}. {d['text']}\n")

    # --------------------------------------------------------

    def analyze(self, image_path):
        out = Path("output5")
        out.mkdir(exist_ok=True)

        dets, img = self.detect(image_path)
        dets = self.ocr_dimensions(dets)

        self.visualize(img, dets, out / "annotated_result.jpg")
        self.save_txt(dets, out / "results.txt")


# ------------------------- RUN -------------------------

if __name__ == "__main__":
    detector = TechnicalDrawingDetector(
        model_path="runs/obb/technical_drawing/weights/best.pt"
    )
    detector.analyze("D:\\projects\project2\\version_3\\dataset-views\\images\\0_100.pdf.jpg")
