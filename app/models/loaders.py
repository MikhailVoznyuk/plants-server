from typing import Any, Dict
import os
from app import config
from ultralytics import YOLO

class LazyModels:
    def __init__(self):
        self.models: Dict[str, Any] = {}

    def _resolve_yolo_weights(self, path: str | None, default_name: str = "yolov8n-seg.pt") -> str:
        p = (path or "").strip()
        return p if (p and os.path.exists(p)) else default_name

    def plant_seg(self):
        if "plant_seg" not in self.models:
            w = self._resolve_yolo_weights(config.PLANT_SEG_WEIGHTS, "yolov8n-seg.pt")
            self.models["plant_seg"] = self._load_yolo_seg(w)
        return self.models["plant_seg"]

    def defect_seg(self):
        if "defect_seg" not in self.models:
            w = self._resolve_yolo_weights(config.DEFECT_SEG_WEIGHTS, "yolov8n-seg.pt")
            self.models["defect_seg"] = self._load_yolo_seg(w)
        return self.models["defect_seg"]

    def species_cls(self):
        if "species_cls" not in self.models:
            self.models["species_cls"] = self._load_species(config.SPECIES_CLS_WEIGHTS)
        return self.models["species_cls"]

    def depth_model(self):
        if "depth" not in self.models:
            self.models["depth"] = self._load_depth_anything_or_midas()
        return self.models["depth"]

    def _load_yolo_seg(self, weights: str):
        model = YOLO(weights)
        model.fuse()
        return model

    def _load_species(self, weights: str):
        if weights and os.path.exists(weights):
            import onnxruntime as ort
            sess = ort.InferenceSession(weights, providers=['CPUExecutionProvider'])
            return ("onnx", sess)
        else:
            return ("stub", None)

    def _load_depth_anything_or_midas(self):
        try:
            from depth_anything.dpt import DepthAnything
            import torch
            m = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14").eval().to(config.DEVICE)
            return ("depth_anything", m)
        except Exception:
            import torch
            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            midas.to(config.DEVICE).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            return ("midas", (midas, transforms))