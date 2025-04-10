import time
import torch
import torchvision
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from IPython.display import HTML
from base64 import b64encode
import torch.nn.functional as F
from queue import Queue
from threading import Thread
from collections import deque  # Pour stocker la trajectoire

from models.tracknet import TrackNet
from utils.general import get_shuttle_position

# Définition des chemins (inspiration YOLOv5)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Répertoire racine
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def parse_opt():
    parser = ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'example_dataset/match/videos/1_10_12.mp4', help='Chemin vers la vidéo.')
    parser.add_argument('--save-txt', action='store_true', help='Enregistrer les résultats dans un fichier CSV')
    parser.add_argument('--view-img', action='store_true', help='Afficher les images')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[288, 512], help='Taille de l’image (hauteur, largeur)')
    parser.add_argument('--weights', type=str, default=ROOT / 'best.pt', help='Chemin vers les poids du modèle entraîné.')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='Dossier pour enregistrer les résultats')
    # Nombre d'échantillons par batch (chaque échantillon contient 3 frames)
    parser.add_argument('--batch-size', type=int, default=1, help='Nombre d’échantillons (chaque échantillon est constitué de 3 frames) traités par batch.')
    opt = parser.parse_args()
    return opt

def show_video(video_path):
    video = open(video_path, "rb").read()
    encoded_video = b64encode(video).decode('ascii')
    return HTML(data=f'''
        <video width="640" height="480" controls>
            <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
        </video>
    ''')

def preprocess_frame(img, target_size):
    # Convertit l'image de BGR à RGB et la redimensionne avec OpenCV
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    return img_resized

def write_frames(video_queue, out):
    """
    Cette fonction s'exécute dans un thread séparé et écrit
    les frames récupérées dans la file d'attente via cv2.VideoWriter.write.
    Un frame de valeur None est utilisé comme signal de terminaison.
    """
    while True:
        frame = video_queue.get()
        if frame is None:
            break
        out.write(frame)
        video_queue.task_done()

def main(opt):
    source_name = os.path.splitext(os.path.basename(opt.source))[0]
    b_save_txt = opt.save_txt
    b_view_img = opt.view_img
    d_save_dir = str(opt.project)
    f_weights = str(opt.weights)
    f_source = str(opt.source)
    imgsz = opt.imgsz  # [hauteur, largeur]
    batch_size = opt.batch_size
    sample_size = 3  # Nombre fixe de frames par échantillon

    # Ajout de "_predict" au nom de la vidéo
    source_name = f"{source_name}_predict"

    # Création des dossiers de sauvegarde
    if not os.path.exists(d_save_dir):
        os.makedirs(d_save_dir)
    img_save_path = os.path.join(d_save_dir, source_name)
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # Chargement et préparation du modèle
    model = TrackNet().to(device)
    model.load_state_dict(torch.load(f_weights, weights_only=True))
    model.eval()

    vid_cap = cv2.VideoCapture(f_source)
    video_end = False

    video_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(d_save_dir, f"{source_name}.mp4"), fourcc, fps, (w, h))

    # Création d'une file d'attente et lancement du thread pour l'écriture asynchrone
    video_queue = Queue()
    writer_thread = Thread(target=write_frames, args=(video_queue, out))
    writer_thread.start()

    if b_save_txt:
        f_save_txt = open(os.path.join(d_save_dir, f"{source_name}.csv"), 'w')
        f_save_txt.write('frame_num,visible,x,y\n')

    # On utilise un deque pour stocker les 10 dernières coordonnées détectées
    trajectory = deque(maxlen=10)

    count = 0
    total_start = time.time()
    total_frames_per_batch = batch_size * sample_size

    while True:
        t0 = time.time()
        imgs = []
        for _ in range(total_frames_per_batch):
            ret, img = vid_cap.read()
            if not ret:
                video_end = True
                break
            imgs.append(img)
        t_read = time.time() - t0

        if len(imgs) == 0:
            break

        # Traitement des échantillons complets uniquement
        complete_frames = (len(imgs) // sample_size) * sample_size
        if complete_frames == 0:
            break
        imgs = imgs[:complete_frames]
        current_total = len(imgs)
        current_batch_size = current_total // sample_size

        # --- Phase de Prétraitement ---
        t0 = time.time()
        frames_np = np.ascontiguousarray(np.stack(imgs, axis=0)[..., ::-1])
        frames_tensor = torch.from_numpy(frames_np).to(device)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2).float().div(255.0)
        resized_frames_tensor = F.interpolate(frames_tensor, size=(imgsz[0], imgsz[1]),
                                              mode='bilinear', align_corners=False)
        frames_tensor_samples = resized_frames_tensor.reshape(current_batch_size, sample_size, 3, imgsz[0], imgsz[1])
        imgs_torch = frames_tensor_samples.reshape(current_batch_size, sample_size * 3, imgsz[0], imgsz[1])
        t_preprocess = time.time() - t0

        # --- Phase d'Inférence ---
        t0 = time.time()
        with torch.no_grad():
            preds = model(imgs_torch)
        t_infer = time.time() - t0

        # --- Phase de Post-traitement ---
        t0 = time.time()
        preds = preds.detach().cpu().numpy()
        y_preds = (preds > 0.5).astype('float32') * 255
        y_preds = y_preds.astype('uint8')
        t_postprocess = time.time() - t0

        print(f"Frame read: {t_read:.4f}s, Preprocessing: {t_preprocess:.4f}s, "
              f"Inference: {t_infer:.4f}s, Postprocessing: {t_postprocess:.4f}s")

        y_preds_split = np.split(y_preds, current_batch_size, axis=0)

        for sample_index, sample in enumerate(y_preds_split):
            sample = sample[0]  # shape: [9, imgsz[0], imgsz[1]]
            sample_splits = np.split(sample, sample_size, axis=0)
            for i in range(sample_size):
                local_idx = sample_index * sample_size + i
                global_frame_index = count + local_idx
                mask = sample_splits[i]
                if len(mask.shape) > 2:
                    mask = mask[0]
                (visible, cx_pred, cy_pred) = get_shuttle_position(mask)
                (cx, cy) = (int(cx_pred * w / imgsz[1]), int(cy_pred * h / imgsz[0]))
                if visible:
                    # Dessiner le cercle de détection (en rouge)
                    cv2.circle(imgs[local_idx], (cx, cy), 6, (0, 0, 255), -1)
                    # Ajouter la coordonnée à la trajectoire
                    trajectory.append((cx, cy))
                # Dessiner la trajectoire (les 10 dernières détections) sur la frame en vert
                for pt in trajectory:
                    cv2.circle(imgs[local_idx], pt, 3, (0, 255, 0), -1)
                if b_save_txt:
                    f_save_txt.write(f'{global_frame_index},{visible},{cx},{cy}\n')
                if b_view_img:
                    plt.imshow(cv2.cvtColor(imgs[local_idx], cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.show()
                # Ajout de la frame dans la file d'attente pour l'écriture asynchrone
                video_queue.put(imgs[local_idx])
                print(f"{global_frame_index} ---- visible: {visible}  cx: {cx}  cy: {cy}")
        count += current_total

        if video_end:
            break

    total_time = time.time() - total_start
    print(f"Total processing time: {total_time:.2f}s")

    if b_save_txt:
        while count < video_len:
            f_save_txt.write(f'{count},0,0,0\n')
            count += 1
        f_save_txt.close()

    vid_cap.release()
    # Signal de terminaison pour le thread d'écriture
    video_queue.put(None)
    writer_thread.join()
    out.release()

    show_video(os.path.join(d_save_dir, f"{source_name}.mp4"))

if __name__ == '__main__':
    opt = parse_opt()
    # Profilage global (optionnel)
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main(opt)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(20)
