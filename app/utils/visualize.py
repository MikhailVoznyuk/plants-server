import cv2
import numpy as np
from typing import List, Dict, Any
from app import config

def overlay_instances(image_bgr, plants, defects, plant_colors=None, defect_colors=None):
    if plant_colors is None: plant_colors = config.PLANT_COLORS
    if defect_colors is None: defect_colors = config.DEFECT_COLORS
    out = image_bgr.copy()

    def color_for(name, table, default):
        return table.get(name, table.get("_default", default))

    for p in plants:
        col = color_for(p["cls"], plant_colors, (0, 200, 0))
        m = p["mask"].astype(bool)
        out[m] = (0.6*out[m] + 0.4*np.array(col, dtype=np.uint8)).astype(np.uint8)
        x1,y1,x2,y2 = p["bbox"]
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
        label = f'{p["cls"]} {p.get("species","")}'.strip()
        cv2.putText(out, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    for d in defects:
        col = color_for(d["cls"], defect_colors, (255, 255, 0))
        m = d["mask"].astype(bool)
        out[m] = (0.6*out[m] + 0.4*np.array(col, dtype=np.uint8)).astype(np.uint8)
        x1,y1,x2,y2 = d["bbox"]
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
        label = f'{d["cls"]} {d.get("severity","")}'.strip()
        cv2.putText(out, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    return out

def compute_tilt_deg(mask: np.ndarray) -> float:
    m = cv2.moments(mask.astype(np.uint8))
    if abs(m["mu20"] + m["mu02"]) < 1e-6:
        return 0.0
    cov = np.array([[m["mu20"], m["mu11"]],
                    [m["mu11"], m["mu02"]]], dtype=np.float64) / max(m["m00"], 1.0)
    eigvals, eigvecs = np.linalg.eig(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    tilt = abs(90 - abs(angle))
    return float(np.clip(tilt, 0.0, 90.0))

def compute_dry_ratio(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    m = mask.astype(bool)
    if m.sum() == 0:
        return 0.0
    dry = (((h >= 5) & (h <= 35) & (s >= 20) & (v >= 40)))
    num = int((dry & m).sum())
    den = int(m.sum())
    return float(num / max(1, den))
