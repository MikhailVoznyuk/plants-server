# app/pipeline.py
import os, uuid, csv, json
from typing import Dict, Any, List, Tuple

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
    def rle_encode(mask):  # fallback
        return None


def _bbox_to_dict(b):
    if isinstance(b, dict):
        return b
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return {"x1": int(b[0]), "y1": int(b[1]), "x2": int(b[2]), "y2": int(b[3])}
    return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}


def _bump(sev: str) -> str:
    order = ["low", "medium", "high"]
    try:
        i = order.index(sev)
        return order[min(i + 1, len(order) - 1)]
    except ValueError:
        return sev


def _grade_from_score(score: float) -> str:
    if score >= 80:
        return "good"
    if score >= 60:
        return "fair"
    if score >= 40:
        return "poor"
    return "critical"


def _auto_thresholds(defects: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_cls: Dict[str, List[float]] = {}
    for d in defects:
        ar = d.get("area_ratio")
        if ar is None:
            continue
        by_cls.setdefault(d["cls"], []).append(float(ar))
    out: Dict[str, Dict[str, float]] = {}
    for cls, vals in by_cls.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size >= 3:
            med = float(np.percentile(arr, 50))
            p85 = float(np.percentile(arr, 85))
        else:
            med, p85 = 0.03, 0.10
        out[cls] = {"median": med, "p85": p85, "n": int(arr.size)}
    return out


def _load_severity_override() -> Dict[str, str] | None:
    path = getattr(config, "SEVERITY_RULES_CSV", "") or ""
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        m: Dict[str, str] = {}
        for _, row in df.iterrows():
            sev = str(row["severity"]).strip().lower()
            if sev not in {"low", "medium", "high"}:
                continue
            m[str(row["defect"]).strip()] = sev
        return m
    except Exception:
        return None


class Pipeline:
    def __init__(self) -> None:
        self.m = LazyModels()

    # ==================== PUBLIC ====================

    def run(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        req_id = str(uuid.uuid4())[:8]
        H, W = image_bgr.shape[:2]

        # A) plants seg
        plants_pred = self._yolo_seg(self.m.plant_seg(), image_bgr, config.THRESH_PLANT)

        # HSV fallback
        if len(plants_pred) == 0 and config.PLANT_FALLBACK:
            veg = self._green_veg_mask(image_bgr)
            plants_pred = self._instances_from_mask(veg, "tree", min_area=5000)

        if len(plants_pred) == 0:
            return {
                "request_id": req_id,
                "status": "NO_PLANTS",
                "overlay_path": None,
                "report_json_path": None,
                "plants": [],
                "defects": [],
                "extras": {},
            }

        # D) depth
        depth = self._depth_map(image_bgr)

        # B) defects seg
        defects_pred = self._yolo_seg(self.m.defect_seg(), image_bgr, config.THRESH_DEFECT)

        # brown fallback (внутри маски растений)
        if len(defects_pred) == 0 and config.DEFECT_FALLBACK:
            union_plants_mask = np.zeros((H, W), np.uint8)
            for p in plants_pred:
                union_plants_mask |= p["mask"].astype(np.uint8)
            brn = self._brown_patch_mask(image_bgr)
            pseudo = cv2.bitwise_and(brn, brn, mask=union_plants_mask)
            defects_pred = self._instances_from_mask(pseudo, "fungus", min_area=800)

        # ROI-clip дефектов маской растений
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

        # Линковка дефект->растение
        self._link_defects_to_plants(clipped_defects, plants_pred, depth)

        # C) species
        self._classify_species(image_bgr, plants_pred)

        # эвристики
        for p in plants_pred:
            p["tilt_deg"] = compute_tilt_deg(p["mask"])
            p["dry_ratio"] = compute_dry_ratio(image_bgr, p["mask"])

        # E+F) правила и здоровье
        out_dir = os.path.join(config.OUT_DIR, req_id)
        os.makedirs(out_dir, exist_ok=True)
        self._apply_rules(plants_pred, clipped_defects, debug=bool(getattr(config, "DEBUG_EXPORT_THR", 0)), out_dir=out_dir)

        # визуализация и экспорт
        overlay = overlay_instances(
            image_bgr, plants_pred, clipped_defects,
            plant_colors=config.PLANT_COLORS, defect_colors=config.DEFECT_COLORS
        )
        overlay_path = os.path.join(out_dir, "overlay.png")
        cv2.imwrite(overlay_path, overlay)

        report = {
            "request_id": req_id,
            "plants": self._serialize_plants(plants_pred),
            "defects": self._serialize_defects(clipped_defects),
        }
        report_json_path = os.path.join(out_dir, "report.json")
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self._save_csv(out_dir, plants_pred, clipped_defects)

        overlay_url = f"/out/{req_id}/overlay.png"
        report_url  = f"/out/{req_id}/report.json"
        
        return {
          "request_id": req_id,
          "status": "OK",
          "overlay_path": overlay_path,
          "report_json_path": report_json_path,
          "plants": report["plants"],
          "defects": report["defects"],
          "extras": {"overlay_url": overlay_url, "report_url": report_url}
        }


    # ==================== INTERNAL ====================

    def _yolo_seg(self, model, image_bgr: np.ndarray, conf_thr: float) -> List[Dict[str, Any]]:
        H, W = image_bgr.shape[:2]
        # Ultralytics ждёт RGB
        results = model.predict(source=image_bgr[..., ::-1], verbose=False, conf=conf_thr, device=config.DEVICE)
        items: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return items
    
        r = results[0]
        # если нет масок — сразу выходим
        if r.masks is None or r.boxes is None or r.masks.data is None:
            return items
    
        # имена классов
        names = r.names if hasattr(r, "names") else {int(i): str(i) for i in np.unique(r.boxes.cls.cpu().numpy().astype(int))}
    
        # забираем тензоры
        masks_np = r.masks.data.cpu().numpy()        # [N, Hm, Wm] — часто НЕ равны [H, W]
        boxes_np = r.boxes.xyxy.cpu().numpy()        # [N, 4] — как правило уже в координатах исходника
        confs_np = r.boxes.conf.cpu().numpy()        # [N]
        cls_np   = r.boxes.cls.cpu().numpy().astype(int)  # [N]
    
        # гарантируем размер масок = (H, W)
        # nearest чтобы не размыть бинарность
        if masks_np.ndim == 3:
            fixed_masks = []
            for m in masks_np:
                if m.shape[0] != H or m.shape[1] != W:
                    m_resized = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                else:
                    m_resized = m
                # бинаризуем в {0,1} как uint8
                fixed_masks.append((m_resized > 0.5).astype(np.uint8))
            masks_np = np.stack(fixed_masks, axis=0)
        else:
            # на всякий случай
            return items
    
        N = masks_np.shape[0]
        for i in range(N):
            mask = masks_np[i]                              # (H, W), uint8 в {0,1}
            area = int(mask.sum())
            x1, y1, x2, y2 = boxes_np[i].astype(int).tolist()
    
            # подрезаем боксы в границы изображения, вдруг YOLO выдал дроби/выход за край
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1
    
            items.append({
                "id": i + 1,
                "cls": names.get(int(cls_np[i]), str(int(cls_np[i]))),
                "conf": float(confs_np[i]),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "area": area,
                "mask": mask,   # гарантия (H, W)
            })
        return items

    def _depth_map(self, image_bgr: np.ndarray) -> np.ndarray:
        typ, obj = self.m.depth_model()
        if typ == "depth_anything":
            model = obj
            im = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (518, 518), interpolation=cv2.INTER_LINEAR)
            im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            im = im.to(config.DEVICE)
            with torch.no_grad():
                pred = model(im)
            depth = pred.squeeze().detach().cpu().numpy()
            depth = cv2.resize(depth, (image_bgr.shape[1], image_bgr.shape[0]))
        else:
            midas, transforms = obj
            tform = transforms.dpt_transform
            im = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            im = tform(im).to(config.DEVICE)
            with torch.no_grad():
                pred = midas(im)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=image_bgr.shape[:2],
                    mode="bicubic",
                    align_corners=False
                ).squeeze()
            depth = pred.detach().cpu().numpy()

        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
        return depth.astype(np.float32)

    def _link_defects_to_plants(self, defects: List[Dict[str, Any]], plants: List[Dict[str, Any]], depth: np.ndarray):
        for d in defects:
            best_score = -1.0
            best_plant = None
            d_mask = d["mask"].astype(bool)
            d_depth_med = np.median(depth[d_mask]) if d_mask.any() else 0.5
            for p in plants:
                p_mask = p["mask"].astype(bool)
                inter = (d_mask & p_mask).sum()
                if inter == 0:
                    continue
                union = d_mask.sum() + p_mask.sum() - inter
                iou = inter / max(1, union)
                p_depth_med = np.median(depth[p_mask]) if p_mask.any() else 0.5
                depth_gap = abs(d_depth_med - p_depth_med)
                score = iou - 0.05 * depth_gap
                if score > best_score:
                    best_score = score
                    best_plant = p
            if best_plant is not None:
                d["plant_id"] = best_plant["id"]

    def _classify_species(self, image_bgr: np.ndarray, plants: List[Dict[str, Any]]):
        kind, sess = self.m.species_cls()
        for p in plants:
            if kind == "onnx":
                x1, y1, x2, y2 = p["bbox"]
                crop = image_bgr[max(0, y1):min(image_bgr.shape[0], y2),
                                 max(0, x1):min(image_bgr.shape[1], x2)]
                if crop.size == 0:
                    p["species"] = "unknown"
                    continue
                im = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
                im = np.transpose(im[..., ::-1], (2, 0, 1))[None, ...]
                try:
                    res = sess.run(None, {"input": im})[0]
                    cls_id = int(np.argmax(res, axis=1)[0])
                    p["species"] = f"class_{cls_id}"
                except Exception:
                    p["species"] = "unknown"
            else:
                p["species"] = config.SPECIES_STUB_LABEL

    def _apply_rules(self, plants: List[Dict[str, Any]], defects: List[Dict[str, Any]],
                     debug: bool = False, out_dir: str | None = None):
        # 1) area_ratio
        plant_by_id = {int(p["id"]): p for p in plants if "id" in p}
        for d in defects:
            pid = int(d.get("plant_id") or next(iter(plant_by_id.keys()), 0))
            pl = plant_by_id.get(pid)
            if not pl:
                d["area_ratio"] = 0.0
                continue
            ar = float(d.get("area", 0)) / max(float(pl.get("area", 1)), 1.0)
            d["area_ratio"] = float(np.clip(ar, 0.0, 1.0))

        # 2) severity: CSV override или авто
        override = _load_severity_override()
        thr = None
        if override:
            for d in defects:
                d["severity"] = override.get(d["cls"], d.get("severity") or "low")
        else:
            thr = _auto_thresholds(defects)
            for d in defects:
                cls = d["cls"]
                ar = float(d.get("area_ratio", 0.0))
                t = thr.get(cls, {"median": 0.03, "p85": 0.10, "n": 0})
                sev = "low" if ar < t["median"] else ("medium" if ar < t["p85"] else "high")
                d["severity"] = sev

        # модификаторы от dry/tilt
        for d in defects:
            pid = int(d.get("plant_id") or next(iter(plant_by_id.keys()), 0))
            pl = plant_by_id.get(pid, {})
            tilt = float(pl.get("tilt_deg", 0.0))
            dry = float(pl.get("dry_ratio", 0.0))
            cls = d["cls"]
            sev = d["severity"]
            if cls in {"fungus", "pests"} and dry > 0.45:
                sev = _bump(sev)
            if cls in {"crack", "cavity", "mech_damage"} and tilt > 20.0:
                sev = _bump(sev)
            d["severity"] = sev

        # 3) health per-plant
        weight = {"cavity": 60, "crack": 45, "mech_damage": 35, "fungus": 30, "pests": 25}
        mult = {"low": 0.7, "medium": 1.0, "high": 1.5}
        penalties = {int(p["id"]): 0.0 for p in plants if "id" in p}

        for d in defects:
            pid = int(d.get("plant_id") or next(iter(plant_by_id.keys()), 0))
            w = float(weight.get(d["cls"], 25))
            m = float(mult.get(d.get("severity") or "low", 1.0))
            ar = float(d.get("area_ratio", 0.0))
            penalties[pid] = penalties.get(pid, 0.0) + w * ar * m

        for p in plants:
            pid = int(p.get("id", 0))
            base = float(penalties.get(pid, 0.0))
            dry = float(p.get("dry_ratio", 0.0))
            tilt = float(p.get("tilt_deg", 0.0))
            base += 0.30 * dry * 100.0
            base += 0.8 * max(0.0, tilt - 10.0)
            score = max(0.0, 100.0 - base)
            p["health_score"] = float(score)
            p["health_grade"] = _grade_from_score(score)

        # 4) debug: сохранить авто-пороги
        if debug and out_dir and thr:
            df = pd.DataFrame(
                [{"defect": k, "median": v["median"], "p85": v["p85"], "n": v["n"]} for k, v in thr.items()]
            ).sort_values("defect")
            df.to_csv(os.path.join(out_dir, "auto_rule_thresholds.csv"), index=False)

    def _serialize_plants(self, plants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for p in plants:
            item = {
                "id": int(p["id"]),
                "cls": p["cls"],
                "conf": float(p.get("conf", 0.0)),
                "bbox": _bbox_to_dict(p.get("bbox")),
                "area": int(p.get("area", 0)),
                "tilt_deg": float(p.get("tilt_deg", 0.0)),
                "dry_ratio": float(p.get("dry_ratio", 0.0)),
                "species": p.get("species"),
                "mask_rle": rle_encode(p.get("mask")) if p.get("mask") is not None else None,
                "health_score": float(p.get("health_score", 100.0)),
                "health_grade": p.get("health_grade"),
            }
            out.append(item)
        return out

    def _serialize_defects(self, defects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for d in defects:
            item = {
                "id": int(d["id"]),
                "cls": d["cls"],
                "conf": float(d.get("conf", 0.0)),
                "bbox": _bbox_to_dict(d.get("bbox")),
                "area": int(d.get("area", 0)),
                "plant_id": int(d.get("plant_id")) if d.get("plant_id") is not None else None,
                "mask_rle": rle_encode(d.get("mask")) if d.get("mask") is not None else None,
                "area_ratio": float(d.get("area_ratio", 0.0)),
                "severity": d.get("severity"),
            }
            out.append(item)
        return out

    def _save_csv(self, out_dir: str, plants: List[Dict[str, Any]], defects: List[Dict[str, Any]]):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "plants.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "id", "cls", "conf", "x1", "y1", "x2", "y2", "area",
                "tilt_deg", "dry_ratio", "species", "health_score", "health_grade"
            ])
            for p in plants:
                bb = _bbox_to_dict(p.get("bbox"))
                w.writerow([
                    p["id"], p["cls"], p.get("conf", 0.0),
                    bb["x1"], bb["y1"], bb["x2"], bb["y2"],
                    p.get("area", 0), p.get("tilt_deg", 0.0), p.get("dry_ratio", 0.0),
                    p.get("species"), p.get("health_score", 100.0), p.get("health_grade")
                ])
        with open(os.path.join(out_dir, "defects.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "id", "cls", "conf", "x1", "y1", "x2", "y2", "area",
                "plant_id", "area_ratio", "severity"
            ])
            for d in defects:
                bb = _bbox_to_dict(d.get("bbox"))
                w.writerow([
                    d["id"], d["cls"], d.get("conf", 0.0),
                    bb["x1"], bb["y1"], bb["x2"], bb["y2"],
                    d.get("area", 0), d.get("plant_id"),
                    d.get("area_ratio", 0.0), d.get("severity")
                ])

    # ==================== HELPERS (HSV/instances) ====================

    def _green_veg_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = (h >= config.VEG_H_MIN) & (h <= config.VEG_H_MAX) & (s >= config.VEG_S_MIN) & (v >= config.VEG_V_MIN)
        mask = mask.astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        return mask

    def _brown_patch_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = (h >= config.BRN_H_MIN) & (h <= config.BRN_H_MAX) & (s >= config.BRN_S_MIN) & (v <= config.BRN_V_MAX)
        return (mask.astype(np.uint8) * 255)

    def _instances_from_mask(self, mask: np.ndarray, cls_name: str, min_area: int = 5000) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
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
                "id": idx, "cls": cls_name, "conf": 0.30,
                "bbox": [x1, y1, x2, y2], "area": area, "mask": comp
            })
            idx += 1
        return out

