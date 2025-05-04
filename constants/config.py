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

# TrackNet ------------------------------------------------------------------
TRACKNET_CKPT      = ROOT_DIR / "weights" / "TrackNet_best.pt"
TRACKNET_CONF      = 0.40      # seuil heatmap
TRACKNET_TRAJ_LEN  = 8         # longueur trajectoire affichée
BALL_CSV_OUT       = ROOT_DIR / "output" / "ball_positions.csv"

DEBUG_TRACKNET     = True      # mettre False pour désactiver les prints



# === Seuils ===
CONF_THRES  = 0.05   # seuil de confiance YOLO

# === Polygones ===


TERRAIN_POLYGON = np.array([
    [652,  65], [608, 104], [454, 249], [298, 439],
    [153, 663], [ 35, 959], [442,1065], [531,1079],
    [1375,1079], [1817, 982], [1693, 681], [1556, 453],
    [1415, 271], [1268, 112], [1228,  72], [1062,  62],
    [856,  58],
], np.int32).reshape((-1, 1, 2))


# TERRAIN_POLYGON = np.array([
#     [700, 176], [658, 215], [508, 377], [269, 701],
#     [221, 779], [132, 971], [ 99,1072], [198,1079],
#     [1852,1078], [1835,1015], [1797, 914], [1744, 799],
#     [1673, 667], [1538, 475], [1480, 398], [1372, 285],
#     [1318, 229], [1281, 188]
# ], np.int32).reshape((-1, 1, 2))


ATTACK_ZONE_1 = np.array([
    [911, 698],[1207, 696],[1690, 679],[1554, 453],
    [1422, 276],[925, 248],[915, 491]
], np.int32).reshape((-1,1,2))



# ATTACK_ZONE_1 = np.array([
#     [657, 214], [606, 268], [509, 375], [651, 374],
#     [821, 370], [1003, 369], [998, 243], [996, 211],
#     [780, 211], [685, 214]
# ], np.int32).reshape((-1, 1, 2))


ATTACK_ZONE_2 = np.array([
    [439, 264],[297, 440],[157, 663],[408, 679],
    [912, 696],[915, 456],[925, 247],[647, 250]
], np.int32).reshape((-1,1,2))

# ATTACK_ZONE_2  = np.array([
#     [996, 212], [999, 270], [1001, 324], [1004, 367],
#     [1142, 378], [1327, 385], [1477, 394], [1368, 277],
#     [1333, 242], [1316, 227], [1266, 224], [1176, 219],
#     [1080, 214], [1014, 212]
# ], np.int32).reshape((-1, 1, 2))





ATTACK_ZONE_3 = np.array([
    [937,  95],[930, 171],[926, 247],[1177,255],
    [1411,268],[1321,167],[1268,113],[1041, 98]
], np.int32).reshape((-1,1,2))

# ATTACK_ZONE_3 = np.array([
#     [504, 380], [441, 453], [376, 543], [307, 640],
#     [222, 778], [435, 805], [714, 827], [1017, 836],
#     [1007, 438], [1002, 369], [808, 367]
# ], np.int32).reshape((-1, 1, 2))

ATTACK_ZONE_4 = np.array([
    [937,  95],[820,  95],[609, 103],[552, 155],
    [450, 256],[647, 248],[927, 248],[930, 171]
], np.int32).reshape((-1,1,2))

# ATTACK_ZONE_4 = np.array([
#     [1005, 370], [1222, 381], [1380, 386], [1478, 393],
#     [1564, 510], [1744, 798], [1448, 824], [1143, 836],
#     [1017, 837], [1013, 632], [1008, 486], [1005, 409]
# ], np.int32).reshape((-1, 1, 2))

ATTACK_ZONES = [
    ATTACK_ZONE_1,
    ATTACK_ZONE_2,
    ATTACK_ZONE_3,
    ATTACK_ZONE_4,
]
