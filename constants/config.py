from pathlib import Path
import numpy as np

# Racine du projet
ROOT_DIR = Path(__file__).resolve().parents[1]

# === Chemins ===
MODEL_PATH  = ROOT_DIR / "weights" / "yolo11m.pt"
VIDEO_IN    = ROOT_DIR / "input"   / "enregistrement.mp4"
VIDEO_OUT   = ROOT_DIR / "output"  / "kpi.mp4"
CSV_OUT     = ROOT_DIR / "output"  / "player_positions.csv"
CSV_DIR     = ROOT_DIR / "output"  / "csv"
KPI_OUT     = CSV_DIR  / "kpi_summary.csv"

# === Seuils ===
CONF_THRES  = 0.3   # seuil de confiance YOLO

# === Polygones ===
TERRAIN_POLYGON = np.array([
    [652,  65], [608, 104], [454, 249], [298, 439],
    [153, 663], [ 35, 959], [442,1065], [531,1079],
    [1375,1079], [1817, 982], [1693, 681], [1556, 453],
    [1415, 271], [1268, 112], [1228,  72], [1062,  62],
    [856,  58],
], np.int32).reshape((-1, 1, 2))

ATTACK_ZONE_1 = np.array([
    [911, 698],[1207, 696],[1690, 679],[1554, 453],
    [1422, 276],[925, 248],[915, 491]
], np.int32).reshape((-1,1,2))

ATTACK_ZONE_2 = np.array([
    [439, 264],[297, 440],[157, 663],[408, 679],
    [912, 696],[915, 456],[925, 247],[647, 250]
], np.int32).reshape((-1,1,2))

ATTACK_ZONE_3 = np.array([
    [937,  95],[930, 171],[926, 247],[1177,255],
    [1411,268],[1321,167],[1268,113],[1041, 98]
], np.int32).reshape((-1,1,2))

ATTACK_ZONE_4 = np.array([
    [937,  95],[820,  95],[609, 103],[552, 155],
    [450, 256],[647, 248],[927, 248],[930, 171]
], np.int32).reshape((-1,1,2))

ATTACK_ZONES = [
    ATTACK_ZONE_1,
    ATTACK_ZONE_2,
    ATTACK_ZONE_3,
    ATTACK_ZONE_4,
]
