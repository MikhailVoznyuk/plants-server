import os, uuid, csv
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import torch

from app import config
from app.models.loaders import LazyModels
from app.utils.visualize import overlay_instances, compute_tilt_deg, compute_dry_ratio

try:
    from app.utils.enc import rle_encode
except Exception:
    def rle_encode(mask):
        return None

class Pipeline:
    def __init__(self) -> None:
        self.m = LazyModels()

    def run(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        req_id = str(uuid.uuid4())[:8]
        H, W = image_bgr.shape[:2]

        plants_pred = self._yolo_seg(self.m.plant_seg(), image_bgr, config.THRESH_PLANT)

        if len(plants_pred) == 0 and config.PLANT_FALLBACK:
            veg = self._green_veg_mask(image_bgr)
            plants_pred = self._instances_from_mask(veg, "tree", min_area=5000)

        if len(plants_pred) == 0:
            return {"request_id": req_id, "status": "NO_PLANTS", "overlay_path": None, "report_json_path": None,
                    "plants": [], "defects": [], "extras": {}}

        depth = self._depth_map(image_bgr)

        defects_pred = self._yolo_seg(self.m.defect_seg(), image_bgr, config.THRESH_DEFECT)

        if len(defects_pred) == 0 and config.DEFECT_FALLBACK:
            union_plants_mask = np.zeros((H, W), np.uint8)
            for p in plants_pred:
                union_plants_mask |= p["mask"].astype(np.uint8)
            brn = self._brown_patch_mask(image_bgr)
            pseudo = cv2.bitwise_and(brn, brn, mask=union_plants_mask)
            defects_pred = self._instances_from_mask(pseudo, "fungus", min_area=800)

        union_plants = np.zeros((H, W), dtype=np.uint8)
        for p in plants_pred:
            union_plants |= p["mask"].astype(np.uint8)

        clipped_defects: List[Dict[str, Any]] = []
        for d in defects_pred:
            m = (d["mask"] > 0) & (union_plants > 0)
            if m.sum() == 0:
                continue
            d2 = d.copy()
            d2["mask"] = m.astype(np.uint8)
            d2["area"] = int(m.sum())
            ys, xs = np.where(m)
            d2["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            clipped_defects.append(d2)

        self._link_defects_to_plants(clipped_defects, plants_pred, depth)

        self._classify_species(image_bgr, plants_pred)

        for p in plants_pred:
            p["tilt_deg"] = compute_tilt_deg(p["mask"])
            p["dry_ratio"] = compute_dry_ratio(image_bgr, p["mask"])

        self._apply_rules(plants_pred, clipped_defects)

        overlay = overlay_instances(image_bgr, plants_pred, clipped_defects,
                                    plant_colors=config.PLANT_COLORS, defect_colors=config.DEFECT_COLORS)
        out_dir = os.path.join(config.OUT_DIR, req_id); os.makedirs(out_dir, exist_ok=True)
        overlay_path = os.path.join(out_dir, "overlay.png"); cv2.imwrite(overlay_path, overlay)

        report = {"request_id": req_id, "plants": self._serialize_plants(plants_pred), "defects": self._serialize_defects(clipped_defects)}
        report_json_path = os.path.join(out_dir, "report.json")
        import json
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self._save_csv(out_dir, plants_pred, clipped_defects)

        return {"request_id": req_id, "status": "OK", "overlay_path": overlay_path, "report_json_path": report_json_path,
                "plants": report["plants"], "defects": report["defects"]}

    def _yolo_seg(self, model, image_bgr: np.ndarray, conf_thr: float) -> List[Dict[str, Any]]:
        H, W = image_bgr.shape[:2]
        results = model.predict(source=image_bgr[..., ::-1], verbose=False, conf=conf_thr, device=config.DEVICE)
        items: List[Dict[str, Any]] = []
        if len(results) == 0:
            return items
        r = results[0]
        names = r.names
        if r.masks is None:
            return items
        masks_small = r.masks.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(masks_small.shape[0]):
            mask = cv2.resize(masks_small[i], (W, H), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.uint8)
            area = int(mask.sum())
            x1, y1, x2, y2 = boxes[i].astype(int).tolist()
            items.append({"id": i+1, "cls": names.get(cls_ids[i], str(cls_ids[i])), "conf": float(confs[i]),
                          "bbox": [x1,y1,x2,y2], "area": area, "mask": mask})
        return items

    def _depth_map(self, image_bgr: np.ndarray) -> np.ndarray:
        typ, obj = self.m.depth_model()
        if typ == "midas":
            midas, transforms = obj
            tform = transforms.dpt_transform
            im = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            im = tform(im).to(config.DEVICE)
            with torch.no_grad():
                pred = midas(im)
                pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=image_bgr.shape[:2],
                                                       mode="bicubic", align_corners=False).squeeze()
            depth = pred.detach().cpu().numpy()
        else:
            depth = np.zeros(image_bgr.shape[:2], np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
        return depth.astype(np.float32)

    def _link_defects_to_plants(self, defects, plants, depth):
        for d in defects:
            best = (-1.0, None, None)
            d_mask = d["mask"].astype(bool)
            d_depth_med = float(np.median(depth[d_mask])) if d_mask.any() else 0.5
            for p in plants:
                p_mask = p["mask"].astype(bool)
                inter = int((d_mask & p_mask).sum())
                if inter == 0: continue
                union = d_mask.sum() + p_mask.sum() - inter
                iou = inter / max(1, union)
                p_depth_med = float(np.median(depth[p_mask])) if p_mask.any() else 0.5
                depth_gap = abs(d_depth_med - p_depth_med)
                score = iou - 0.05 * depth_gap
                if score > best[0]:
                    best = (score, p, depth_gap)
            if best[1] is not None:
                d["plant_id"] = best[1]["id"]

    def _classify_species(self, image_bgr, plants):
        kind, sess = self.m.species_cls()
        for p in plants:
            if kind == "onnx":
                x1,y1,x2,y2 = p["bbox"]
                crop = image_bgr[max(0,y1):min(image_bgr.shape[0],y2), max(0,x1):min(image_bgr.shape[1],x2)]
                if crop.size == 0:
                    p["species"] = "unknown"; continue
                im = cv2.resize(crop, (224,224)).astype(np.float32)/255.0
                im = np.transpose(im[..., ::-1], (2,0,1))[None, ...]
                try:
                    res = sess.run(None, {"input": im})[0]
                    import numpy as np
                    cls_id = int(np.argmax(res, axis=1)[0])
                    p["species"] = f"class_{cls_id}"
                except Exception:
                    p["species"] = "unknown"
            else:
                p["species"] = config.SPECIES_STUB_LABEL

    def _apply_rules(self, plants, defects, debug: bool=False, out_dir: str|None=None):
        plant_area = {int(p["id"]): float(p["mask"].sum()) for p in plants}
        for d in defects:
            pid = d.get("plant_id", None); p_area = plant_area.get(int(pid), 0.0) if pid is not None else 0.0
            d["area_ratio"] = float(d["area"] / max(1.0, p_area))

        thresholds = None
        applied = False
        path = getattr(config, "SEVERITY_RULES_CSV", "") or ""
        if path and os.path.exists(path):
            try:
                df = pd.read_csv(path)
                rule = {str(r["defect"]).strip(): str(r["severity"]).strip().lower() for _, r in df.iterrows()}
                for d in defects:
                    sev = rule.get(d["cls"], None)
                    if sev in {"low","medium","high"}:
                        d["severity"] = sev
                applied = True
            except Exception:
                applied = False

        if not applied:
            by_class = {}
            for d in defects:
                by_class.setdefault(d["cls"], []).append(float(d.get("area_ratio", 0.0)))
            thresholds = {}
            for cls, arr in by_class.items():
                import numpy as np
                arr = np.array(arr, dtype=float)
                if arr.size >= 3:
                    med = float(np.percentile(arr, 50))
                    p85 = float(np.percentile(arr, 85))
                else:
                    med, p85 = 0.03, 0.10
                thresholds[cls] = {"median": med, "p85": p85, "n": int(arr.size)}
            for d in defects:
                t = thresholds.get(d["cls"], {"median":0.03,"p85":0.10})
                ar = float(d.get("area_ratio", 0.0))
                sev = "low" if ar < t["median"] else ("medium" if ar < t["p85"] else "high")
                d["severity"] = sev

        plant_by_id = {int(p["id"]): p for p in plants}
        def _bump(sev: str) -> str:
            order = ["low","medium","high"]
            try:
                i = order.index(sev); return order[min(i+1, 2)]
            except ValueError:
                return sev

        for d in defects:
            pid = d.get("plant_id", None)
            pl = plant_by_id.get(int(pid), {}) if pid is not None else {}
            tilt = float(pl.get("tilt_deg", 0.0))
            dry  = float(pl.get("dry_ratio", 0.0))
            sev = d["severity"]
            cls = d["cls"]
            if cls in {"fungus","pests"} and dry > 0.45:
                sev = _bump(sev)
            if cls in {"crack","cavity","mech_damage"} and tilt > 20.0:
                sev = _bump(sev)
            d["severity"] = sev

        weight = {"cavity":60.0, "crack":45.0, "mech_damage":35.0, "fungus":30.0, "pests":25.0}
        mult = {"low":0.7, "medium":1.0, "high":1.5}
        acc = {int(p["id"]): 0.0 for p in plants}
        for d in defects:
            pid = d.get("plant_id", None)
            if pid is None: continue
            w = float(weight.get(d["cls"], 25.0))
            m = float(mult.get(d.get("severity") or "low", 1.0))
            ar = float(d.get("area_ratio", 0.0))
            acc[int(pid)] = acc.get(int(pid), 0.0) + w * ar * m

        for p in plants:
            pid = int(p["id"])
            base = float(acc.get(pid, 0.0))
            base += 0.30 * float(p.get("dry_ratio", 0.0)) * 100.0
            base += 0.8 * max(0.0, float(p.get("tilt_deg", 0.0)) - 10.0)
            score = max(0.0, 100.0 - base)
            p["health_score"] = float(round(score, 1))
            p["health_grade"] = ("good" if score>=80 else "fair" if score>=60 else "poor" if score>=40 else "critical")

        if debug and out_dir and thresholds:
            df = pd.DataFrame([{"defect": k, "median": v["median"], "p85": v["p85"], "n": v["n"]} for k,v in thresholds.items()]).sort_values("defect")
            df.to_csv(os.path.join(out_dir, "auto_rule_thresholds.csv"), index=False)

    # helpers exposed
    def _green_veg_mask(self, img): return self.__class__._green_veg_mask_static(img)
    @staticmethod
    def _green_veg_mask_static(image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = ((h >= config.VEG_H_MIN) & (h <= config.VEG_H_MAX) & (s >= config.VEG_S_MIN) & (v >= config.VEG_V_MIN))
        mask = mask.astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        return mask

    def _brown_patch_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = ((h >= config.BRN_H_MIN) & (h <= config.BRN_H_MAX) & (s >= config.BRN_S_MIN) & (v <= config.BRN_V_MAX))
        return (mask.astype(np.uint8) * 255)

    def _instances_from_mask(self, mask: np.ndarray, cls_name: str, min_area: int=5000):
        out = []
        num, lbl = cv2.connectedComponents(mask)
        idx = 1
        for i in range(1, num):
            comp = (lbl == i).astype(np.uint8)
            area = int(comp.sum())
            if area < min_area: continue
            ys, xs = np.where(comp)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            out.append({"id": idx, "cls": cls_name, "conf": 0.30, "bbox": [x1,y1,x2,y2], "area": area, "mask": comp})
            idx += 1
        return out

    def _overlay(self, img, plants, defects):
        return overlay_instances(img, plants, defects)

    def _compute_tilt_on_mask(self, mask):
        return compute_tilt_deg(mask)

    def _compute_dry_on_mask(self, img, mask):
        return compute_dry_ratio(img, mask)
