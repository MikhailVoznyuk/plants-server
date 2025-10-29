import os
from ultralytics import YOLO
import onnxruntime as ort
import torch

from app import config

class LazyModels:
    def __init__(self) -> None:
        self._plant = None
        self._defect = None
        self._species = None
        self._depth = None

    def plant_seg(self):
        if self._plant is None:
            path = config.WEIGHTS_PLANT
            if os.path.exists(path):
                self._plant = YOLO(path)
            else:
                self._plant = YOLO("yolov8n-seg.pt")
        return self._plant

    def defect_seg(self):
        if self._defect is None:
            path = config.WEIGHTS_DEFECT
            if os.path.exists(path):
                self._defect = YOLO(path)
            else:
                self._defect = YOLO("yolov8n-seg.pt")
        return self._defect

    def species_cls(self):
        if self._species is None:
            if config.SPECIES_ONNX and os.path.exists(config.SPECIES_ONNX):
                sess = ort.InferenceSession(config.SPECIES_ONNX, providers=["CPUExecutionProvider"])
                self._species = ("onnx", sess)
            else:
                self._species = ("stub", None)
        return self._species

    def depth_model(self):
        if self._depth is None:
            device = torch.device(config.DEVICE)
            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            midas.to(device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self._depth = ("midas", (midas, transforms))
        return self._depth
