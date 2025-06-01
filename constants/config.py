from pathlib import Path
import numpy as  np
import os



from dotenv import load_dotenv
import os

load_dotenv()  # ← Obligatoire pour charger .env dans os.environ

API_BASE_URL = os.getenv("API_BASE_URL")
LOGIN_CREDENTIALS = {
    "login": os.getenv("LOGIN"),
    "password": os.getenv("PASSWORD")
}




class Config:
    s3_enabled = True
    s3_key = os.getenv("S3_ACCESS_KEY")
    s3_secret = os.getenv("S3_SECRET_KEY")
    s3_bucket = os.getenv("S3_BUCKET_NAME")
    s3_endpoint = os.getenv("S3_ENDPOINT")
    s3_public_endpoint = os.getenv("S3_PUBLIC_ENDPOINT")

# Racine du projet
ROOT_DIR = Path(__file__).resolve().parents[1]




# === Chemins ===
MODEL_PATH  = ROOT_DIR / "weights" / "yolo11m.pt"
VIDEO_IN    = ROOT_DIR / "input"   / "fillet.mp4" 


# VIDEO_IN    = ROOT_DIR / "Enregistrement"   /"recordings"/ "match_2025.mp4" 
VIDEO_OUT   = ROOT_DIR / "output"  / "kpi.mp4"
CSV_OUT     = ROOT_DIR / "output"  / "player_positions.csv"
CSV_DIR     = ROOT_DIR / "output"  / "csv"
KPI_OUT     = CSV_DIR  / "kpi_summary.csv"

# TrackNet ------------------------------------------------------------------
TRACKNET_CKPT      = ROOT_DIR / "weights" / "TrackNet_best.pt"
TRACKNET_CONF      = 0.3      # seuil heatmap
TRACKNET_TRAJ_LEN  = 8         # longueur trajectoire affichée
BALL_CSV_OUT       = ROOT_DIR / "output" / "ball_positions.csv"

DEBUG_TRACKNET     = True      # mettre False pour désactiver les prints


# Définissez vos couleurs distinctes pour chaque zone d'attaque (BGR)
attack_zone_colors = [
    (0, 0, 255),    # Rouge
    (0, 255, 255),  # Jaune
    (255, 0, 0),    # Bleu
    (0, 255, 0),    # Vert
]



# === Seuils ===
CONF_THRES  = 0.05   # seuil de confiance YOLO

# === Polygones ===


# TERRAIN_POLYGON = np.array([
#     [652,  65], [608, 104], [454, 249], [298, 439],
#     [153, 663], [ 35, 959], [442,1065], [531,1079],
#     [1375,1079], [1817, 982], [1693, 681], [1556, 453],
#     [1415, 271], [1268, 112], [1228,  72], [1062,  62],
#     [856,  58],
# ], np.int32).reshape((-1, 1, 2))
# import numpy as np

TERRAIN_POLYGON = np.array([
    [652, 50],
    [605, 89],
    [443, 245],
    [293, 422],
    [151, 645],
    [68, 836],
    [35, 947],
    [674, 1079],
    [1239, 1078],
    [1807, 976],
    [1769, 861],
    [1637, 580],
    [1546, 440],
    [1412, 261],
    [1314, 152],
    [1267, 101],
    [1227, 63],
    [1001, 47],
    [746, 47],
], np.int32).reshape((-1, 1, 2))




# ATTACK_ZONE_2 = np.array([
#     [911, 698],[1207, 696],[1690, 679],[1554, 453],
#     [1422, 276],[925, 248],[915, 491]
# ], np.int32).reshape((-1,1,2))


# ATTACK_ZONE_1 = np.array([
#     [439, 264],[297, 440],[157, 663],[408, 679],
#     [912, 696],[915, 456],[925, 247],[647, 250]
# ], np.int32).reshape((-1,1,2))


# ATTACK_ZONE_3 = np.array([
#     [937,  95],[930, 171],[926, 247],[1177,255],
#     [1411,268],[1321,167],[1268,113],[1041, 98]
# ], np.int32).reshape((-1,1,2))

# ATTACK_ZONE_4 = np.array([
#     [937,  95],[820,  95],[609, 103],[552, 155],
#     [450, 256],[647, 248],[927, 248],[930, 171]
# ], np.int32).reshape((-1,1,2))


# TERRAIN_POLYGON7 = np.array([
#     [700, 176], [658, 215], [508, 377], [269, 701],
#     [221, 779], [132, 971], [ 99,1072], [198,1079],
#     [1852,1078], [1835,1015], [1797, 914], [1744, 799],
#     [1673, 667], [1538, 475], [1480, 398], [1372, 285],
#     [1318, 229], [1281, 188]
# ], np.int32).reshape((-1, 1, 2))

# ATTACK_ZONE7_1 = np.array([
#     [657, 214], [606, 268], [509, 375], [651, 374],
#     [821, 370], [1003, 369], [998, 243], [996, 211],
#     [780, 211], [685, 214]
# ], np.int32).reshape((-1, 1, 2))

# ATTACK_ZONE7_2  = np.array([
#     [996, 212], [999, 270], [1001, 324], [1004, 367],
#     [1142, 378], [1327, 385], [1477, 394], [1368, 277],
#     [1333, 242], [1316, 227], [1266, 224], [1176, 219],
#     [1080, 214], [1014, 212]
# ], np.int32).reshape((-1, 1, 2))

# ATTACK_ZONE7_3 = np.array([
#     [504, 380], [441, 453], [376, 543], [307, 640],
#     [222, 778], [435, 805], [714, 827], [1017, 836],
#     [1007, 438], [1002, 369], [808, 367]
# ], np.int32).reshape((-1, 1, 2))

# ATTACK_ZONE7_4 = np.array([
#     [1005, 370], [1222, 381], [1380, 386], [1478, 393],
#     [1564, 510], [1744, 798], [1448, 824], [1143, 836],
#     [1017, 837], [1013, 632], [1008, 486], [1005, 409]
# ], np.int32).reshape((-1, 1, 2))




ATTACK_ZONE_1 = np.array([
    [151, 646],
    [448, 668],
    [904, 683],
    [914, 391],
    [923, 236],
    [707, 237],
    [442, 246],
    [196, 562],
], np.int32).reshape((-1, 1, 2))

ATTACK_ZONE_2 = np.array([
    [904, 684],
    [1258, 685],
    [1606, 674],
    [1683, 672],
    [1550, 445],
    [1472, 339],
    [1415, 262],
    [1204, 249],
    [1001, 240],
    [923, 237],
    [912, 390],
    [903, 609],
], np.int32).reshape((-1, 1, 2))

ATTACK_ZONE_3 = np.array([
    [922, 236],
    [1410, 258],
    [1316, 152],
    [1267, 102],
    [1027, 85],
    [934, 82],
    [928, 155],
], np.int32).reshape((-1, 1, 2))

ATTACK_ZONE_4 = np.array([
    [922, 236],
    [928, 168],
    [936, 82],
    [779, 82],
    [641, 87],
    [606, 89],
    [552, 136],
    [444, 244],
    [671, 239],
    [812, 238],
], np.int32).reshape((-1, 1, 2))



ATTACK_ZONES = [
    ATTACK_ZONE_1,
    ATTACK_ZONE_2,
    ATTACK_ZONE_3,
    ATTACK_ZONE_4,
]


# ZONE_POLYGONS = {
#     1: np.array([
#         [439, 265],
#         [295, 440],
#         [203, 582],
#         [128, 720],
#         [76, 849],
#         [45, 964],
#         [285, 1031],
#         [561, 1079],
#         [912, 1079],
#         [910, 697],
#         [916, 448],
#         [926, 248],
#         [806, 248],
#         [640, 249]
#     ], dtype=np.int32),

    

#     2:np.array([
#     [926, 248],
#     [918, 426],
#     [911, 575],
#     [912, 698],
#     [912, 1077],
#     [1291, 1077],
#     [1584, 1038],
#     [1817, 982],
#     [1781, 871],
#     [1692, 679],
#     [1556, 456],
#     [1421, 274],
#     [1251, 263],
#     [1037, 251]
#     ], dtype=np.int32),



#     3: np.array([
#         [937, 58],
#         [937, 94],
#         [931, 174],
#         [926, 246],
#         [1074, 251],
#         [1326, 261],
#         [1409, 262],
#         [1318, 164],
#         [1268, 112],
#         [1228, 73],
#         [1128, 66],
#         [1004, 59]
#     ], dtype=np.int32),

#     4: np.array([
#         [653, 66],
#         [772, 59],
#         [939, 58],
#         [936, 95],
#         [930, 177],
#         [926, 248],
#         [739, 248],
#         [614, 250],
#         [447, 255],
#         [546, 159],
#         [609, 103]
#     ], dtype=np.int32),

# }




ZONE_POLYGONS = {
    1: np.array([
        [436, 249],
        [294, 422],
        [149, 648],
        [67, 833],
        [37, 948],
        [673, 1078],
        [902, 1078],
        [904, 682],
        [910, 413],
        [922, 236],
        [787, 237],
        [598, 242],
    ], dtype=np.int32),

    2: np.array([
        [922, 236],
        [1153, 246],
        [1414, 260],
        [1547, 441],
        [1684, 671],
        [1771, 861],
        [1804, 975],
        [1229, 1078],
        [907, 1078],
        [904, 683],
        [914, 354],
    ], dtype=np.int32),

    3: np.array([
        [923, 236],
        [929, 154],
        [936, 82],
        [937, 46],
        [1073, 51],
        [1226, 61],
        [1268, 101],
        [1335, 172],
        [1412, 259],
        [1256, 250],
        [1051, 240],
    ], dtype=np.int32),

    4: np.array([
        [923, 236],
        [930, 147],
        [936, 82],
        [939, 46],
        [795, 45],
        [653, 49],
        [605, 89],
        [543, 145],
        [445, 243],
        [598, 239],
        [792, 236],
    ], dtype=np.int32),
}



NET_POLY = np.array([
    [447, 247],
    [677, 244],
    [924, 238],
    [1059, 248],
    [1226, 255],
    [1417, 268],
    [1431, 213],
    [1146, 191],
    [927, 186],
    [702, 184],
    [496, 193],
    [431, 197]
], dtype=np.int32).reshape((-1, 1, 2))


