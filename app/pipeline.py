import os, uuid, cv2, numpy as np, torch, pandas as pd
from typing import Dict, Any, List
from app import config
from app.models.loaders import LazyModels
from app.utils.visualize import overlay_instances, compute_tilt_deg, compute_dry_ratio
from app.utils.enc import rle_encode

def to_device(arr: torch.Tensor):
    return arr.to(config.DEVICE) if isinstance(arr, torch.Tensor) else arr

def _green_veg_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (h >= config.VEG_H_MIN) & (h <= config.VEG_H_MAX) & (s >= config.VEG_S_MIN) & (v >= config.VEG_V_MIN)
    mask = mask.astype(np.uint8) * 255
    # немного морфологии, чтобы убрать шум
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

def _brown_patch_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (h >= config.BRN_H_MIN) & (h <= config.BRN_H_MAX) & (s >= config.BRN_S_MIN) & (v <= config.BRN_V_MAX)
    return (mask.astype(np.uint8) * 255)

def _instances_from_mask(mask: np.ndarray, cls_name: str, min_area: int = 5000) -> List[Dict[str, Any]]:
    out = []
    num, lbl = cv2.connectedComponents(mask)
    idx = 1
    for i in range(1, num):
        comp = (lbl == i).astype(np.uint8)
        area = int(comp.sum())
        if area < min_area:
            continue
        ys, xs = np.where(comp)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        out.append({
            "id": idx,
            "cls": cls_name,
            "conf": 0.30,
            "bbox": [x1, y1, x2, y2],
            "area": area,
            "mask": comp
        })
        idx += 1
    return out

class Pipeline:
    def __init__(self): self.m = LazyModels()

    def run(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        req_id = str(uuid.uuid4())[:8]; H, W = image_bgr.shape[:2]
        plants_pred = self._yolo_seg(self.m.plant_seg(), image_bgr, config.THRESH_PLANT)
        if len(plants_pred) == 0 and config.PLANT_FALLBACK:
            veg = _green_veg_mask(image_bgr)
            plants_pred = _instances_from_mask(veg, "tree", min_area=5000)

        if len(plants_pred) == 0:
            return {
                "request_id": req_id,
                "status": "NO_PLANTS",
                "plants": [],
                "defects": [],
                "extras": {}
            }
        depth = self._depth_map(image_bgr)
        defects_pred = self._yolo_seg(self.m.defect_seg(), image_bgr, config.THRESH_DEFECT)
        if len(defects_pred) == 0 and config.DEFECT_FALLBACK:
            # коричневые зоны внутри зелени считаем псевдо-дефектами
            union_plants_mask = np.zeros(image_bgr.shape[:2], np.uint8)
            for p in plants_pred:
                union_plants_mask |= p["mask"].astype(np.uint8)
            brn = _brown_patch_mask(image_bgr)
            pseudo = cv2.bitwise_and(brn, brn, mask=union_plants_mask)
            defects_pred = _instances_from_mask(pseudo, "fungus", min_area=800)  # имя класса на твой вкус

        union_plants = np.zeros((H, W), dtype=np.uint8)
        for p in plants_pred: union_plants |= p["mask"].astype(np.uint8)
        clipped_defects = []
        for d in defects_pred:
            m = (d["mask"] > 0) & (union_plants > 0)
            if m.sum() == 0: continue
            d2 = d.copy(); d2["mask"] = m.astype(np.uint8); d2["area"] = int(m.sum())
            ys, xs = np.where(m); d2["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            clipped_defects.append(d2)
        self._link_defects_to_plants(clipped_defects, plants_pred, depth)
        self._classify_species(image_bgr, plants_pred)
        for p in plants_pred:
            p["tilt_deg"] = compute_tilt_deg(p["mask"]); p["dry_ratio"] = compute_dry_ratio(image_bgr, p["mask"])
        self._apply_rules(plants_pred, clipped_defects)
        overlay = overlay_instances(image_bgr, plants_pred, clipped_defects, plant_colors=config.PLANT_COLORS, defect_colors=config.DEFECT_COLORS)
        out_dir = os.path.join(config.OUT_DIR, req_id); os.makedirs(out_dir, exist_ok=True)
        overlay_path = os.path.join(out_dir, "overlay.png"); cv2.imwrite(overlay_path, overlay)
        report = {"request_id": req_id, "plants": self._serialize_plants(plants_pred), "defects": self._serialize_defects(clipped_defects)}
        report_json_path = os.path.join(out_dir, "report.json")
        with open(report_json_path, "w", encoding="utf-8") as f:
            import json; json.dump(report, f, ensure_ascii=False, indent=2)
        self._save_csv(out_dir, plants_pred, clipped_defects)
        return {"request_id": req_id, "status": "OK", "overlay_path": overlay_path, "report_json_path": report_json_path, "plants": report["plants"], "defects": report["defects"]}

    def _yolo_seg(self, model, image_bgr, conf_thr) -> List[Dict[str, Any]]:
        results = model.predict(source=image_bgr[..., ::-1], verbose=False, conf=conf_thr, device=config.DEVICE)
        items = []; 
        if len(results) == 0: return items
        r = results[0]; names = r.names
        if r.masks is None: return items
        masks = r.masks.data.cpu().numpy(); boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy(); cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(masks.shape[0]):
            mask = (masks[i] > 0.5).astype(np.uint8); area = int(mask.sum())
            x1,y1,x2,y2 = boxes[i].astype(int).tolist()
            items.append({"id": i+1, "cls": names.get(cls_ids[i], str(cls_ids[i])), "conf": float(confs[i]), "bbox": [x1,y1,x2,y2], "area": area, "mask": mask})
        return items

    def _depth_map(self, image_bgr: np.ndarray) -> np.ndarray:
        typ, obj = self.m.depth_model()
        if typ == "depth_anything":
            model = obj; import torch, cv2
            im = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB); im = cv2.resize(im, (518,518), interpolation=cv2.INTER_LINEAR)
            im = torch.from_numpy(im).permute(2,0,1).unsqueeze(0).float()/255.0; im = im.to(config.DEVICE)
            with torch.no_grad(): pred = model(im)
            depth = pred.squeeze().detach().cpu().numpy()
            depth = cv2.resize(depth, (image_bgr.shape[1], image_bgr.shape[0]))
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9); return depth.astype(np.float32)
        else:
            midas, transforms = obj; import torch, cv2
            tform = transforms.dpt_transform; im = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB); im = tform(im).to(config.DEVICE)
            with torch.no_grad():
                pred = midas(im)
                pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=image_bgr.shape[:2], mode="bicubic", align_corners=False).squeeze()
            depth = pred.detach().cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9); return depth.astype(np.float32)

    def _link_defects_to_plants(self, defects, plants, depth):
        for d in defects:
            best = (-1.0, None, None); d_mask = d["mask"].astype(bool)
            d_depth_med = np.median(depth[d_mask]) if d_mask.any() else 0.5
            for p in plants:
                p_mask = p["mask"].astype(bool); inter = (d_mask & p_mask).sum()
                if inter == 0: continue
                union = d_mask.sum() + p_mask.sum() - inter; iou = inter / max(1, union)
                p_depth_med = np.median(depth[p_mask]) if p_mask.any() else 0.5
                depth_gap = abs(d_depth_med - p_depth_med); score = iou - 0.05*depth_gap
                if score > best[0]: best = (score, p, depth_gap)
            if best[1] is not None: d["plant_id"] = best[1]["id"]

    def _classify_species(self, image_bgr, plants):
        kind, sess = self.m.species_cls()
        for p in plants:
            if kind == "onnx":
                x1,y1,x2,y2 = p["bbox"]
                crop = image_bgr[max(0,y1):min(image_bgr.shape[0],y2), max(0,x1):min(image_bgr.shape[1],x2)]
                if crop.size == 0: p["species"] = "unknown"; continue
                import cv2, numpy as np
                im = cv2.resize(crop, (224,224)).astype(np.float32)/255.0; im = np.transpose(im[..., ::-1], (2,0,1))[None, ...]
                try:
                    res = sess.run(None, {"input": im})[0]; cls_id = int(np.argmax(res, axis=1)[0]); p["species"] = f"class_{cls_id}"
                except Exception: p["species"] = "unknown"
            else:
                p["species"] = config.SPECIES_STUB_LABEL

    def _apply_rules(self, plants, defects):
        import numpy as np, os, pandas as pd
        plant_area = {p["id"]: float(p["mask"].sum()) for p in plants}
        for d in defects:
            pid = d.get("plant_id", None); p_area = plant_area.get(pid, 0.0) if pid is not None else 0.0
            d["area_ratio"] = float(d["area"] / max(1.0, p_area))

        applied_from_csv = False
        if config.SEVERITY_RULES_CSV and os.path.exists(config.SEVERITY_RULES_CSV):
            try:
                df = pd.read_csv(config.SEVERITY_RULES_CSV)
                rule = {str(r["defect"]).strip(): str(r["severity"]).strip() for _, r in df.iterrows()}
                for d in defects: d["severity"] = rule.get(d["cls"], None)
                applied_from_csv = True
            except Exception: applied_from_csv = False

        if not applied_from_csv:
            by_class = {}
            for d in defects: by_class.setdefault(d["cls"], []).append(d["area_ratio"])
            thresholds = {}
            for cls, arr in by_class.items():
                arr = np.array(arr, dtype=float)
                if arr.size == 0: thresholds[cls] = (0.05, 0.15)
                else:
                    med = float(np.quantile(arr, 0.5)); hi = float(np.quantile(arr, 0.85))
                    med = med if med > 1e-6 else 0.05; hi = max(hi, med + 0.05); thresholds[cls] = (med, hi)
            def bump(level, k=1): return int(max(0, min(2, level + k)))
            for d in defects:
                t_med, t_hi = thresholds.get(d["cls"], (0.05, 0.15)); ar = float(d["area_ratio"])
                if ar >= t_hi: lvl = 2
                elif ar >= t_med: lvl = 1
                else: lvl = 0
                pid = d.get("plant_id"); p = next((x for x in plants if x["id"] == pid), None) if pid is not None else None
                dry = float(p.get("dry_ratio", 0.0)) if p else 0.0; tilt = float(p.get("tilt_deg", 0.0)) if p else 0.0
                if d["cls"] in ("fungus","pests") and dry > 0.45: lvl = bump(lvl, 1)
                if d["cls"] in ("crack","cavity","mech_damage") and tilt > 20.0: lvl = bump(lvl, 1)
                d["severity"] = ("low","medium","high")[lvl]

        base_weights = {"cavity": 60.0, "crack": 45.0, "mech_damage": 35.0, "fungus": 30.0, "pests": 25.0, "_default": 25.0}
        mult = {"low": 0.7, "medium": 1.0, "high": 1.5}
        defects_by_plant = {}
        for d in defects:
            pid = d.get("plant_id"); 
            if pid is None: continue
            defects_by_plant.setdefault(pid, []).append(d)

        for p in plants:
            penalties = 0.0
            for d in defects_by_plant.get(p["id"], []):
                w = base_weights.get(d["cls"], base_weights["_default"]); sev = d.get("severity","medium"); m = mult.get(sev,1.0); ar = float(d.get("area_ratio",0.0))
                penalties += w * ar * m
            penalties += 0.30 * float(p.get("dry_ratio",0.0)) * 100.0
            penalties += 0.8 * max(0.0, float(p.get("tilt_deg",0.0)) - 10.0)
            health = max(0.0, 100.0 - penalties); p["health_score"] = float(round(health,1))
            if health >= 80: p["health_grade"] = "good"
            elif health >= 60: p["health_grade"] = "fair"
            elif health >= 40: p["health_grade"] = "poor"
            else: p["health_grade"] = "critical"

    def _serialize_plants(self, plants):
        out = []
        for p in plants:
            x1,y1,x2,y2 = p["bbox"]
            out.append({"id": p["id"], "cls": p["cls"], "conf": float(p["conf"]), "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}, "area": int(p["area"]), "tilt_deg": float(p.get("tilt_deg",0.0)), "dry_ratio": float(p.get("dry_ratio",0.0)), "species": p.get("species", None), "mask_rle": rle_encode(p["mask"]), "health_score": float(p.get("health_score",100.0)), "health_grade": p.get("health_grade", None)})
        return out

    def _serialize_defects(self, defects):
        out = []
        for d in defects:
            x1,y1,x2,y2 = d["bbox"]
            out.append({"id": d["id"], "cls": d["cls"], "conf": float(d["conf"]), "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}, "area": int(d["area"]), "plant_id": int(d["plant_id"]) if d.get("plant_id") is not None else None, "mask_rle": rle_encode(d["mask"]), "area_ratio": float(d.get("area_ratio",0.0)), "severity": d.get("severity", None)})
        return out

    def _save_csv(self, out_dir, plants, defects):
        import pandas as pd
        p_rows = []
        for p in plants:
            p_rows.append({"plant_id": p["id"], "class": p["cls"], "conf": p["conf"], "area": p["area"], "bbox": p["bbox"], "tilt_deg": p.get("tilt_deg",0.0), "dry_ratio": p.get("dry_ratio",0.0), "species": p.get("species", None), "health_score": p.get("health_score",100.0), "health_grade": p.get("health_grade", None)})
        d_rows = []
        for d in defects:
            d_rows.append({"defect_id": d["id"], "class": d["cls"], "conf": d["conf"], "area": d["area"], "bbox": d["bbox"], "plant_id": d.get("plant_id", None), "severity": d.get("severity", None), "area_ratio": d.get("area_ratio", 0.0)})
        if p_rows: pd.DataFrame(p_rows).to_csv(os.path.join(out_dir, "plants.csv"), index=False)
        if d_rows: pd.DataFrame(d_rows).to_csv(os.path.join(out_dir, "defects.csv"), index=False)