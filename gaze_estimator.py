"""
Estimativa do ponto de olhar usando MediaPipe Face Mesh.

Pipeline interno:
  1. Conversão BGR → RGB
  2. Detecção de malha facial (468 landmarks + 10 de íris)
  3. Extração dos landmarks das íris (esquerda: 468, direita: 473)
  4. Cálculo da posição normalizada da íris dentro do olho
  5. Retorno do vetor de olhar médio (ambos os olhos)

A posição normalizada é calculada como:
  norm_x = (iris_center_x - eye_center_x) / (eye_width  / 2)
  norm_y = (iris_center_y - eye_center_y) / (eye_height / 2)

Valores em [-1, 1] onde:
  norm_x < 0 → olhando para a esquerda (do ponto de vista da câmera)
  norm_x > 0 → olhando para a direita
  norm_y < 0 → olhando para cima
  norm_y > 0 → olhando para baixo

Referências:
  Krafka et al. (2016) – iTracker, aprendizado profundo para gaze
  Zhang et al. (2015)  – Gaze estimation in the wild
"""

import os
import cv2
import numpy as np
import urllib.request

from config import (LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
                    LEFT_EYE_OUTER, LEFT_EYE_INNER,
                    LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                    RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
                    RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)

# Detecta qual API o MediaPipe instalado oferece
import mediapipe as mp

_USE_LEGACY = False
try:
    _test = mp.solutions.face_mesh   # AttributeError se API nova
    _USE_LEGACY = True
    print('[GazeEstimator] API: mp.solutions (legacy)')
except AttributeError:
    print('[GazeEstimator] API: FaceLandmarker Tasks (nova)')

# Modelo para a Tasks API
_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
_MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
               'face_landmarker/face_landmarker/float16/1/face_landmarker.task')

def _ensure_model():
    """Baixa o modelo se ainda não existir localmente."""
    if os.path.exists(_MODEL_PATH):
        return
    print(f'[GazeEstimator] Baixando modelo (~6 MB)...')
    print(f'  URL: {_MODEL_URL}')
    try:
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print('[GazeEstimator] Modelo salvo em:', _MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            f'Falha ao baixar o modelo MediaPipe: {e}\n'
            f'Baixe manualmente de:\n  {_MODEL_URL}\n'
            f'e salve como: {_MODEL_PATH}'
        )


# Implementação unificada

class GazeEstimator:
    """
    Detecta face e estima o vetor de olhar via posição da íris.

    Funciona tanto com a API legacy (mp.solutions.face_mesh)
    quanto com a nova Tasks API (FaceLandmarker), detectando
    automaticamente qual está disponível.
    """

    def __init__(self):
        if _USE_LEGACY:
            self._init_legacy()
        else:
            self._init_tasks()

    # Inicialização

    def _init_legacy(self):
        """Inicializa usando mp.solutions.face_mesh."""
        _mp = mp.solutions.face_mesh
        self._face_mesh = _mp.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = None

    def _init_tasks(self):
        """Inicializa usando a Tasks API (FaceLandmarker)."""
        _ensure_model()
        from mediapipe.tasks import python as _mpt
        from mediapipe.tasks.python import vision as _mpv

        base_opts = _mpt.BaseOptions(model_asset_path=_MODEL_PATH)
        opts = _mpv.FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector  = _mpv.FaceLandmarker.create_from_options(opts)
        self._face_mesh = None

    # Processamento de frame

    def process(self, frame_bgr: np.ndarray) -> dict | None:
        """
        Processa um frame BGR e retorna as features de olhar.

        Returns dict com gaze_vector, iris_left_px, etc.
        Returns None se nenhuma face for detectada.
        """
        if _USE_LEGACY:
            return self._process_legacy(frame_bgr)
        else:
            return self._process_tasks(frame_bgr)

    def _process_legacy(self, frame_bgr: np.ndarray) -> dict | None:
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res  = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark  # 478 pontos

        def px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=float)

        return self._compute_gaze(px, w, h)

    def _process_tasks(self, frame_bgr: np.ndarray) -> dict | None:
        h, w = frame_bgr.shape[:2]
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_img)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]  # List[NormalizedLandmark], 478 pontos

        # Verifica se tem iris landmarks (índice 468+)
        if len(lm) < 478:
            print(f'[GazeEstimator] AVISO: modelo retornou apenas {len(lm)} '
                  f'landmarks (sem íris). Use o modelo full, não o lite.')
            return None

        def px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=float)

        return self._compute_gaze(px, w, h)

    # Cálculo do vetor de olhar (igual para ambas as APIs)

    def _compute_gaze(self, px_fn, w: int, h: int) -> dict:
        """
        Calcula o gaze_vector normalizado a partir da posição da íris.
        px_fn(idx) → np.array([x_pixel, y_pixel])
        """
        # Olho esquerdo
        iris_l      = px_fn(LEFT_IRIS_CENTER)
        eye_l_inner = px_fn(LEFT_EYE_INNER)
        eye_l_outer = px_fn(LEFT_EYE_OUTER)
        eye_l_top   = px_fn(LEFT_EYE_TOP)
        eye_l_bot   = px_fn(LEFT_EYE_BOTTOM)

        eye_l_w      = np.linalg.norm(eye_l_outer - eye_l_inner)
        eye_l_h      = np.linalg.norm(eye_l_bot   - eye_l_top)
        eye_l_center = (eye_l_inner + eye_l_outer) / 2

        if eye_l_w > 1 and eye_l_h > 1:
            lnx = (iris_l[0] - eye_l_center[0]) / (eye_l_w / 2)
            lny = (iris_l[1] - eye_l_center[1]) / (eye_l_h / 2)
            ear_l = eye_l_h / eye_l_w
        else:
            lnx, lny = 0.0, 0.0
            ear_l = 0.3  

        # Olho direito
        iris_r      = px_fn(RIGHT_IRIS_CENTER)
        eye_r_inner = px_fn(RIGHT_EYE_INNER)
        eye_r_outer = px_fn(RIGHT_EYE_OUTER)
        eye_r_top   = px_fn(RIGHT_EYE_TOP)
        eye_r_bot   = px_fn(RIGHT_EYE_BOTTOM)

        eye_r_w      = np.linalg.norm(eye_r_outer - eye_r_inner)
        eye_r_h      = np.linalg.norm(eye_r_bot   - eye_r_top)
        eye_r_center = (eye_r_inner + eye_r_outer) / 2

        if eye_r_w > 1 and eye_r_h > 1:
            rnx = (iris_r[0] - eye_r_center[0]) / (eye_r_w / 2)
            rny = (iris_r[1] - eye_r_center[1]) / (eye_r_h / 2)
            ear_r = eye_r_h / eye_r_w
        else:
            rnx, rny = 0.0, 0.0
            ear_r = 0.3 

        gaze_x = (lnx + rnx) / 2
        gaze_y = (lny + rny) / 2

        return {
            'gaze_vector':     (gaze_x, gaze_y),
            'left_iris_norm':  (lnx, lny),
            'right_iris_norm': (rnx, rny),
            'ear':             (ear_l + ear_r) / 2,
            'iris_left_px':    iris_l.astype(int),
            'iris_right_px':   iris_r.astype(int),
            'eye_l_center_px': eye_l_center.astype(int),
            'eye_r_center_px': eye_r_center.astype(int),
        }

    # Debug visual

    def draw_debug(self, frame: np.ndarray, feat: dict | None) -> np.ndarray:
        """Desenha overlays de diagnóstico no frame."""
        if feat is None:
            cv2.putText(frame, 'Nenhuma face detectada', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        cv2.circle(frame, tuple(feat['iris_left_px']),  6, (0, 255, 80), -1)
        cv2.circle(frame, tuple(feat['iris_right_px']), 6, (0, 255, 80), -1)
        cv2.circle(frame, tuple(feat['eye_l_center_px']), 3, (255, 100, 0), -1)
        cv2.circle(frame, tuple(feat['eye_r_center_px']), 3, (255, 100, 0), -1)

        gx, gy = feat['gaze_vector']
        ear    = feat.get('ear', 0)
        cv2.putText(frame, f'Gaze: ({gx:+.3f}, {gy:+.3f})  EAR: {ear:.3f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return frame

    def close(self):
        if self._face_mesh:
            self._face_mesh.close()
