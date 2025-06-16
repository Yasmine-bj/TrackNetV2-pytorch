# fault_net_flash.py  ────────────────────────────────────────────────
import numpy as np, cv2
from collections import deque

class FaultNetDetector:
    """
    Détecteur de faute filet
    + segment rouge temporaire (flash)
    + attribution d’équipe basée sur le point juste avant l’entrée.
    """

    def __init__(self, net_poly, fps,
                 m_stuck=5, gap_s=0.8, gap_tol=3,
                 n_back=5, flash_len_s=1.0):
        """
        net_poly   : np.array(N,1,2)  polygone surface filet
        fps        : images/seconde
        m_stuck    : frames mini coincées ⇒ faute
        gap_s      : délai mini entre deux fautes (anti-doublon)
        gap_tol    : frames manquées tolérées pendant l’observation
        n_back     : nb de points avant la faute affichés en rouge
        flash_len_s: durée d’affichage du segment rouge (sec)
        """
        self.net_poly   = net_poly.astype(np.int32)
        self.fps        = fps
        self.M          = m_stuck
        self.min_gap    = int(gap_s * fps)
        self.gap_tol    = gap_tol
        self.n_back     = n_back
        self.flash_len  = int(flash_len_s * fps)

        # ligne médiane horizontale du filet
        self.y_net = np.median(net_poly[:, 0, 1])

        # — état suiveur
        self.observing   = False
        self.inside_cnt  = 0
        self.missed_cnt  = 0
        self.entry_k     = -1
        self.entry_camp  = None
        self.last_fault  = -self.min_gap

        # — mémoire des centres
        self.recent_pts      = deque(maxlen=n_back)      # [(x,y), …]
        self.prev_visible_pt = None                     # (x,y) hors filet
        self.active_flashes  = []                       # [{pts, ttl}, …]

        # — historique permanent pour KPI
        self.fault_history   = []                       # [(camp, frame)]

    # ------------------------------------------------ utilitaires ----
    def inside(self, x, y):
        return cv2.pointPolygonTest(
            self.net_poly, (float(x), float(y)), False) >= 0

    def camp_of(self, y):
        """-1 = équipe bas (IDs 1-2), +1 = équipe haut (IDs 3-4)."""
        return -1 if y < self.y_net else 1

    # ------------------------------------------------ update ----------
    def update(self, k, x, y, visible=True):
        """
        k, x, y  : numéro de frame + centre balle (px)
        visible  : True si TrackNet a détecté la balle
        Retourne True si faute confirmée À CETTE frame.
        """
        if visible:
            self.recent_pts.append((x, y))
            self.prev_visible_pt = (x, y)              # mémorise dernier point vu

        inside = visible and self.inside(x, y)

        # 1) entrée dans la zone filet --------------------------------
        if inside and not self.observing:
            # camp d'origine = point juste avant l'entrée s'il existe
            if self.prev_visible_pt is not None:
                _, prev_y = self.prev_visible_pt
                self.entry_camp = self.camp_of(prev_y)
            else:
                self.entry_camp = self.camp_of(y)

            self.observing   = True
            self.inside_cnt  = 1
            self.missed_cnt  = 0
            self.entry_k     = k
            return False

        # 2) observation en cours -------------------------------------
        if self.observing:
            if inside:
                self.inside_cnt += 1
                self.missed_cnt  = 0
                return False
            else:
                self.missed_cnt += 1
                if self.missed_cnt <= self.gap_tol:
                    return False          # bruit toléré

                # sortie définitive : décision
                self.observing = False
                if (self.inside_cnt >= self.M and
                    k - self.last_fault > self.min_gap):

                    self.last_fault = k

                    # a) enregistrer la faute
                    self.fault_history.append((self.entry_camp, k))

                    # b) créer le flash rouge
                    seg = list(self.recent_pts)[-self.n_back:]
                    if seg:
                        self.active_flashes.append(
                            {"pts": seg, "ttl": self.flash_len}
                        )

                    print(f"[FAULT] frame {k}  camp "
                          f"{'bas' if self.entry_camp==-1 else 'haut'}  "
                          f"(inside_cnt={self.inside_cnt})")
                    return True
        return False

    # --------------------------------------------- dessin flash -------
    def draw_fault_segments(self, frame_bgr,
                            color=(0,0,255), radius=4):
        """Dessine les segments rouges actifs puis décrémente leur TTL."""
        survivors = []
        for flash in self.active_flashes:
            for x, y in flash["pts"]:
                cv2.circle(frame_bgr, (int(x), int(y)), radius, color, -1)
            flash["ttl"] -= 1
            if flash["ttl"] > 0:
                survivors.append(flash)
        self.active_flashes = survivors
        return frame_bgr

    # --------------------------------------------- KPI ---------------
    def summary(self):
        team2 = sum(1 for camp, _ in self.fault_history if camp == -1)
        team1 = sum(1 for camp, _ in self.fault_history if camp ==  1)
        return team1, team2
