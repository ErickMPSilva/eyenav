"""
Estimativa do ponto de olhar usando MediaPipe Face Mesh.

Pipeline interno:
  1. Conversão BGR → RGB
  2. Detecção de malha facial (468 landmarks + 10 de íris)
  3. Centro da íris = média dos 5 landmarks da íris (centro + contorno)
  4. Posição normalizada da íris dentro do olho
  5. Retorno do vetor de olhar médio (ambos os olhos) e do EAR

Normalização (ambos os eixos pela LARGURA do olho):
  norm_x = (iris_x - eye_center_x) / (eye_width / 2)
  norm_y = (iris_y - eye_center_y) / (eye_width / 2)

A largura (canto interno → canto externo) é estável. A altura do olho
varia com a pálpebra a cada microfechamento; usá-la no denominador de
norm_y amplificava o ruído vertical, já que o olho mede poucos pixels
de altura na imagem.

Média de 5 landmarks da íris: reduz o ruído de quantização de um único
ponto, significativo quando o olho ocupa ~30 px na imagem.

Vetor de features (6 valores, cada olho separado):
  [lnx, lny, rnx, rny, l_lid, r_lid]

  lid = (pálpebra_superior_y − centro_do_olho_y) / (largura / 2)

A íris se desloca pouco na vertical (±0,05 em norm_y contra ±0,15 na
horizontal), então o eixo Y sozinho é fraco e ruidoso. A pálpebra
superior acompanha o olhar vertical — sobe ao olhar para cima, desce
ao olhar para baixo — e fornece um segundo sinal vertical, mais
estável. Manter os olhos separados, em vez da média, permite ao
modelo aproveitar a vergência e compensar um olho com landmark pior.

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

# Landmarks da íris: centro + 4 pontos do contorno
_LEFT_IRIS  = list(range(LEFT_IRIS_CENTER,  LEFT_IRIS_CENTER + 5))    # 468–472
_RIGHT_IRIS = list(range(RIGHT_IRIS_CENTER, RIGHT_IRIS_CENTER + 5))   # 473–477

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
    print('[GazeEstimator] Baixando modelo (~6 MB)...')
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


class GazeEstimator:
    """
    Detecta face e estima o vetor de olhar via posição da íris.

    Funciona tanto com a API legacy (mp.solutions.face_mesh)
    quanto com a nova Tasks API (FaceLandmarker).
    """

    def __init__(self):
        if _USE_LEGACY:
            self._init_legacy()
        else:
            self._init_tasks()

    # Inicialização

    def _init_legacy(self):
        _mp = mp.solutions.face_mesh
        self._face_mesh = _mp.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = None

    def _init_tasks(self):
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
        Processa um frame BGR e retorna as features de olhar,
        ou None se nenhuma face for detectada.
        """
        if _USE_LEGACY:
            return self._process_legacy(frame_bgr)
        return self._process_tasks(frame_bgr)

    def _process_legacy(self, frame_bgr: np.ndarray) -> dict | None:
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res  = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark

        def px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=float)

        return self._compute_gaze(px)

    def _process_tasks(self, frame_bgr: np.ndarray) -> dict | None:
        h, w = frame_bgr.shape[:2]
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_img)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        if len(lm) < 478:
            print(f'[GazeEstimator] AVISO: modelo retornou apenas {len(lm)} '
                  f'landmarks (sem íris). Use o modelo full, não o lite.')
            return None

        def px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=float)

        return self._compute_gaze(px)

    # Cálculo do vetor de olhar

    @staticmethod
    def _eye(px_fn, iris_ids, inner, outer, top, bottom):
        """
        Retorna (norm_x, norm_y, lid, ear, iris_px, center_px) de um olho.
        """
        iris   = np.mean([px_fn(i) for i in iris_ids], axis=0)
        p_in   = px_fn(inner)
        p_out  = px_fn(outer)
        p_top  = px_fn(top)
        eye_w  = np.linalg.norm(p_out - p_in)
        eye_h  = np.linalg.norm(px_fn(bottom) - p_top)
        center = (p_in + p_out) / 2

        if eye_w <= 1:
            return 0.0, 0.0, 0.0, 0.3, iris, center

        half_w = eye_w / 2
        nx  = (iris[0] - center[0]) / half_w
        ny  = (iris[1] - center[1]) / half_w
        lid = (p_top[1] - center[1]) / half_w
        ear = eye_h / eye_w
        return nx, ny, lid, ear, iris, center

    def _compute_gaze(self, px_fn) -> dict:
        lnx, lny, lid_l, ear_l, iris_l, c_l = self._eye(
            px_fn, _LEFT_IRIS, LEFT_EYE_INNER, LEFT_EYE_OUTER,
            LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
        rnx, rny, lid_r, ear_r, iris_r, c_r = self._eye(
            px_fn, _RIGHT_IRIS, RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
            RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)

        return {
            'features':        (lnx, lny, rnx, rny, lid_l, lid_r),
            'gaze_vector':     ((lnx + rnx) / 2, (lny + rny) / 2),
            'left_iris_norm':  (lnx, lny),
            'right_iris_norm': (rnx, rny),
            'ear':             (ear_l + ear_r) / 2,
            'iris_left_px':    iris_l.astype(int),
            'iris_right_px':   iris_r.astype(int),
            'eye_l_center_px': c_l.astype(int),
            'eye_r_center_px': c_r.astype(int),
        }

    # Debug visual

    def draw_debug(self, frame: np.ndarray, feat: dict | None) -> np.ndarray:
        """Desenha overlays de diagnóstico no frame."""
        if feat is None:
            cv2.putText(frame, 'Nenhuma face detectada', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        cv2.circle(frame, tuple(feat['iris_left_px']),  4, (0, 255, 80), -1)
        cv2.circle(frame, tuple(feat['iris_right_px']), 4, (0, 255, 80), -1)
        cv2.circle(frame, tuple(feat['eye_l_center_px']), 2, (255, 100, 0), -1)
        cv2.circle(frame, tuple(feat['eye_r_center_px']), 2, (255, 100, 0), -1)
        return frame

    def close(self):
        if self._face_mesh:
            self._face_mesh.close()