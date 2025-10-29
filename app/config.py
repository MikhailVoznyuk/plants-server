import os

DEVICE = os.getenv("DEVICE", "cpu")
OUT_DIR = os.getenv("OUT_DIR", "/data/out")

THRESH_PLANT = float(os.getenv("THRESH_PLANT", "0.25"))
THRESH_DEFECT = float(os.getenv("THRESH_DEFECT", "0.25"))

PLANT_FALLBACK = os.getenv("PLANT_FALLBACK", "1") == "1"
DEFECT_FALLBACK = os.getenv("DEFECT_FALLBACK", "1") == "1"

VEG_H_MIN = int(os.getenv("VEG_H_MIN", "25"))
VEG_H_MAX = int(os.getenv("VEG_H_MAX", "95"))
VEG_S_MIN = int(os.getenv("VEG_S_MIN", "25"))
VEG_V_MIN = int(os.getenv("VEG_V_MIN", "25"))

BRN_H_MIN = int(os.getenv("BRN_H_MIN", "5"))
BRN_H_MAX = int(os.getenv("BRN_H_MAX", "25"))
BRN_S_MIN = int(os.getenv("BRN_S_MIN", "20"))
BRN_V_MAX = int(os.getenv("BRN_V_MAX", "180"))

WEIGHTS_PLANT = os.getenv("WEIGHTS_PLANT", "/weights/plant_seg.pt")
WEIGHTS_DEFECT = os.getenv("WEIGHTS_DEFECT", "/weights/defect_seg.pt")
SPECIES_ONNX = os.getenv("SPECIES_ONNX", "")
SPECIES_STUB_LABEL = os.getenv("SPECIES_STUB_LABEL", "unknown")

SEVERITY_RULES_CSV = os.getenv("SEVERITY_RULES_CSV", "")

PLANT_COLORS = {
    "tree": (0, 220, 0),
    "shrub": (0, 180, 60),
    "_default": (0, 200, 0)
}
DEFECT_COLORS = {
    "fungus": (0, 180, 255),
    "pests": (0, 120, 255),
    "crack": (0, 0, 255),
    "cavity": (255, 0, 0),
    "mech_damage": (255, 120, 0),
    "_default": (255, 255, 0)
}

DEPTH_BACKEND = os.getenv("DEPTH_BACKEND", "midas")
