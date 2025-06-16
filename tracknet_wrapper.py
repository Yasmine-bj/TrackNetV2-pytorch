"""
TrackNet wrapper
────────────────
- Construit l'entrée attendue pour bg_mode='concat'
  (median RGB + seq_len × RGB  →  (seq_len+1)×3 canaux)
- Renvoie (bx, by, visibility) par frame
- Stocke et dessine la trajectoire (longueur TRACKNET_TRAJ_LEN)
- Si DEBUG_TRACKNET=True dans constants/config.py, affiche temps d'inférence
  et bbox de la balle à chaque appel.
"""

import sys, pathlib, collections, cv2, torch, numpy as np, time
from constants.config import (
    TRACKNET_CKPT, TRACKNET_CONF, TRACKNET_TRAJ_LEN, DEBUG_TRACKNET
)

# ── rendre tracknet_core importable ──────────────────────────────────────
CORE_DIR = pathlib.Path(__file__).resolve().parent / "tracknet_core"
sys.path.append(str(CORE_DIR))

from utils.general import get_model, HEIGHT, WIDTH, predict_location


class TrackNetWrapper:
    def __init__(self, ckpt_path=TRACKNET_CKPT, device="cuda:0"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt      = torch.load(ckpt_path, map_location=device)
        self.mode = ckpt["param_dict"]["bg_mode"]          # 'concat'
        self.seq  = ckpt["param_dict"]["seq_len"]          # 8

        self.net  = get_model("TrackNet", self.seq, self.mode).to(device)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()

        self.device = device
        self.buf    = collections.deque(maxlen=self.seq)            # 8 frames RGB
        self.traj   = collections.deque(maxlen=TRACKNET_TRAJ_LEN)    # stocke (pt, color)
        self.median_rgb = None                                      # calc. 1×

        if DEBUG_TRACKNET:
            print(f"[DBG] TrackNet device={device}  seq={self.seq}  mode={self.mode}")

    # ------------------------------------------------------------------ #
    def _prep(self, frames_rgb):
        """Prépare l'entrée concat : median RGB + seq × RGB."""
        frames = [cv2.resize(f, (WIDTH, HEIGHT)) for f in frames_rgb]

        # médiane unique
        if self.median_rgb is None:
            self.median_rgb = np.median(np.stack(frames), axis=0).astype(np.uint8)
            if DEBUG_TRACKNET:
                print("[DBG] median image initialisée")

        if self.mode == "concat":
            median = self.median_rgb.transpose(2, 0, 1)              # (3,H,W)
            rgbs   = [f.transpose(2, 0, 1) for f in frames]           # L×(3,H,W)
            stack  = np.concatenate([[median], rgbs], axis=0)         # (L+1,3,H,W)
        else:                                                        # RGB pur
            stack  = np.stack([f.transpose(2,0,1) for f in frames])   # (L,3,H,W)

        stack = stack.reshape(1, -1, HEIGHT, WIDTH).astype(np.float32) / 255.0
        return torch.from_numpy(stack).to(self.device)

    # ------------------------------------------------------------------ #
    def update(self, frame_bgr):
        """Ajoute la frame, renvoie (bx, by, vis)."""
        self.buf.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if len(self.buf) < self.seq:
            self.traj.append(None)
            if DEBUG_TRACKNET:
                print(f"[DBG] buffer {len(self.buf)}/{self.seq}")
            return None, None, 0.0

        t0 = time.perf_counter()
        with torch.no_grad():
            heat = self.net(self._prep(self.buf))[0, -1]
        infer_ms = (time.perf_counter() - t0) * 1000

        heat = (heat > TRACKNET_CONF).cpu().numpy()
        x1, y1, w, h = predict_location((heat * 255).astype(np.uint8))

        if DEBUG_TRACKNET:
            print(f"[DBG] infer {infer_ms:5.1f} ms  bbox ({x1},{y1},{w},{h})  vis={w>0}")

        if w == h == 0:
            self.traj.append(None)
            return None, None, 0.0

        H, W, _ = frame_bgr.shape
        bx = int((x1 + w/2) * W / WIDTH)
        by = int((y1 + h/2) * H / HEIGHT)
        self.traj.append((bx, by))
        return bx, by, 1.0
    # ------------------------------------------------------------------ #
    def draw_traj(self, frame_bgr,color):
        for pt in self.traj:
                if pt is not None:
                    cv2.circle(frame_bgr, pt, 3 ,color, -1)
        return frame_bgr
