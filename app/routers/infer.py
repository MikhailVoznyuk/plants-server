import os
import uuid
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.pipeline import Pipeline
from app import config

router = APIRouter()
pipe = Pipeline()

def _read_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Cannot decode image")
    return img

@router.post("/infer", tags=["inference"], summary="Infer")
async def infer(file: UploadFile = File(...)):
    img = _read_image(file)
    res = pipe.run(img)
    return res

@router.post("/debug/depth", tags=["debug"], summary="Depth")
async def debug_depth(file: UploadFile = File(...)):
    img = _read_image(file)
    req_id = str(uuid.uuid4())[:8]
    depth = pipe._depth_map(img)
    out_dir = os.path.join(config.OUT_DIR, req_id)
    os.makedirs(out_dir, exist_ok=True)
    d = (depth * 255).astype(np.uint8)
    path = os.path.join(out_dir, "D_depth.png")
    cv2.imwrite(path, d)
    return {"request_id": req_id, "status": "OK", "depth_png_path": path}

@router.post("/debug/heuristics", tags=["debug"], summary="Heuristics")
async def debug_heuristics(file: UploadFile = File(...)):
    img = _read_image(file)
    req_id = str(uuid.uuid4())[:8]
    mask = pipe._green_veg_mask(img)
    tilt = pipe._compute_tilt_on_mask(mask)
    dry  = pipe._compute_dry_on_mask(img, mask)
    out_dir = os.path.join(config.OUT_DIR, req_id)
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "veg_mask.png")
    cv2.imwrite(mpath, mask)
    return {"request_id": req_id, "status": "OK", "mask_png_path": mpath, "tilt_deg": round(float(tilt), 2), "dry_ratio": round(float(dry), 3)}

@router.post("/debug/export-smoke", tags=["debug"], summary="Export Smoke")
async def export_smoke(file: UploadFile = File(...)):
    img = _read_image(file)
    H, W = img.shape[:2]
    req_id = str(uuid.uuid4())[:8]

    plant_mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(plant_mask, (W//2, H//2), (W//4, H//3), 0, 0, 360, 255, -1)
    plants = [{
        "id": 1, "cls": "tree", "conf": 0.99,
        "bbox": [W//2 - W//4, H//2 - H//3, W//2 + W//4, H//2 + H//3],
        "area": int(plant_mask.sum()),
        "mask": plant_mask,
        "tilt_deg": 8.0,
        "dry_ratio": 0.35,
    }]

    defect_mask = np.zeros((H, W), np.uint8)
    cv2.circle(defect_mask, (W//2 + W//8, H//2), min(W, H)//14, 255, -1)
    defects = [{
        "id": 1, "cls": "fungus", "conf": 0.8,
        "bbox": [W//2, H//2 - H//14, W//2 + W//4, H//2 + H//14],
        "area": int(defect_mask.sum()),
        "mask": defect_mask,
        "plant_id": 1
    }]

    out_dir = os.path.join(config.OUT_DIR, req_id)
    os.makedirs(out_dir, exist_ok=True)

    pipe._apply_rules(plants, defects, debug=True, out_dir=out_dir)
    overlay = pipe._overlay(img, plants, defects)
    opath = os.path.join(out_dir, "overlay.png")
    cv2.imwrite(opath, overlay)

    rep = {"request_id": req_id, "plants": pipe._serialize_plants(plants), "defects": pipe._serialize_defects(defects)}
    rpath = os.path.join(out_dir, "report.json")
    import json
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    pipe._save_csv(out_dir, plants, defects)
    return {"request_id": req_id, "status": "OK", "overlay_path": opath, "report_json_path": rpath}
