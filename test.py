import cv2
from constants.config import (
    ATTACK_ZONES,
    ZONE_POLYGONS,
    NET_POLY 
)
import csv


img = cv2.imread('input/too.png')  # ton image de référence





# Dessiner le polygone du filet (NET_POLY) en rouge
cv2.polylines(img, [NET_POLY], isClosed=True, color=(0, 0, 255), thickness=3)
# Afficher l'étiquette "NET" au centre du polygone
M = cv2.moments(NET_POLY)
if M['m00'] != 0:
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(img, 'NET', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv2.LINE_AA)



for zone_id, polygon in ZONE_POLYGONS.items():
    cv2.polylines(img, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
    # afficher l’ID au centre du polygone
    M = cv2.moments(polygon)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(img, f'Zone {zone_id}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)

# Dessiner les zones d’attaque (ATTACK_ZONES) en bleu
for idx, attack_polygon in enumerate(ATTACK_ZONES, start=1):
    cv2.polylines(img, [attack_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
    # afficher l’étiquette au centre du polygone
    M = cv2.moments(attack_polygon)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(img, f'Atk {idx}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv2.LINE_AA)


cv2.imwrite('output/zones_debug2.jpg', img)
