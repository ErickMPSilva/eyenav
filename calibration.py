"""
Tela de calibração visual em janela OpenCV fullscreen.

Exibe uma grade de CALIB_ROWS × CALIB_COLS pontos.
Para cada ponto:
  1. Usuário fixa o olhar no círculo.
  2. Sistema coleta CALIB_SAMPLES vetores de olhar.
  3. Calcula a média (reduz ruído) e registra o par (gaze, tela).

A sequência de pontos percorre a grade em ordem de leitura
(esquerda → direita, cima → baixo), com margem configurável
para evitar os cantos extremos da tela onde a precisão costuma
ser menor.

Tecla ESC durante a calibração cancela a operação.
"""

import cv2
import numpy as np
import time
from config import (CALIB_ROWS, CALIB_COLS, CALIB_MARGIN_X,
                    CALIB_MARGIN_Y, CALIB_SAMPLES)


# Helpers

def build_calibration_points(sw: int, sh: int) -> list[tuple[int, int]]:
    pts = []
    for row in range(CALIB_ROWS):
        for col in range(CALIB_COLS):
            x = int(sw * (CALIB_MARGIN_X +
                    (1 - 2 * CALIB_MARGIN_X) * col / max(CALIB_COLS - 1, 1)))
            y = int(sh * (CALIB_MARGIN_Y +
                    (1 - 2 * CALIB_MARGIN_Y) * row / max(CALIB_ROWS - 1, 1)))
            pts.append((x, y))
    return pts


# Classe principal
class CalibrationScreen:
    """
    Janela de calibração visual.
    """

    WIN = 'EyeNav – Calibração'

    def __init__(self, sw: int, sh: int):
        self.sw     = sw
        self.sh     = sh
        self.points = build_calibration_points(sw, sh)

    def run(self, estimator, cap) -> tuple[list, list]:
        """
        Executa o processo completo de calibração.

        Parameters
        ----------
        estimator : GazeEstimator
        cap       : cv2.VideoCapture (deve estar aberta)

        Returns
        -------
        features : list of (gaze_x, gaze_y)
        targets  : list of (screen_x, screen_y)
            Listas vazias se o usuário cancelar com ESC.
        """
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.WIN, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        all_features: list = []
        all_targets:  list = []

        for idx, pt in enumerate(self.points):
            result = self._collect_point(idx, pt, estimator, cap)
            if result is None:            # ESC pressionado
                cv2.destroyWindow(self.WIN)
                return [], []
            all_features.append(result)
            all_targets.append(pt)

        # Tela de conclusão
        done_canvas = self._blank()
        cv2.putText(done_canvas, 'Calibração concluída!',
                    (self.sw // 2 - 200, self.sh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)
        cv2.imshow(self.WIN, done_canvas)
        cv2.waitKey(1200)
        cv2.destroyWindow(self.WIN)

        return all_features, all_targets

    # Privado

    def _collect_point(self, idx: int, pt: tuple,
                       estimator, cap) -> tuple | None:
        """
        Coleta amostras para um único ponto de calibração.

        Exibe contagem regressiva visual (arco que preenche).
        Retorna a média das amostras ou None se cancelado.
        """
        samples: list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            feat  = estimator.process(frame)

            canvas = self._blank()
            self._draw_instructions(canvas, idx)
            self._draw_all_points(canvas, idx, len(samples))
            self._draw_active(canvas, pt, len(samples))

            # Coleta amostra
            if feat is not None:
                samples.append(feat['gaze_vector'])

            cv2.imshow(self.WIN, canvas)

            if len(samples) >= CALIB_SAMPLES:
                # Flash verde = ponto concluído
                ok = self._blank()
                self._draw_all_points(ok, idx + 1, CALIB_SAMPLES)
                cv2.circle(ok, pt, 28, (0, 255, 80), -1)
                cv2.imshow(self.WIN, ok)
                cv2.waitKey(350)
                # Média para reduzir ruído pontual
                return tuple(np.mean(samples, axis=0))

            if cv2.waitKey(1) & 0xFF == 27:
                return None

    def _blank(self) -> np.ndarray:
        return np.zeros((self.sh, self.sw, 3), dtype=np.uint8)

    def _draw_instructions(self, canvas: np.ndarray, idx: int):
        msg = (f'Ponto {idx + 1} de {len(self.points)}'
               f'  —  Fixe o olhar no círculo até completar o arco')
        cv2.putText(canvas, msg,
                    (self.sw // 2 - len(msg) * 5, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (180, 180, 180), 1)
        cv2.putText(canvas, 'ESC = cancelar',
                    (self.sw - 170, self.sh - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    def _draw_all_points(self, canvas: np.ndarray, done: int, _samples: int):
        """Desenha todos os pontos — concluídos e futuros."""
        for i, p in enumerate(self.points):
            if i < done:
                cv2.circle(canvas, p, 10, (60, 60, 60), -1)   # concluído
            elif i > done:
                cv2.circle(canvas, p, 10, (60, 60, 60), 1)    # futuro

    def _draw_active(self, canvas: np.ndarray, pt: tuple, n_samples: int):
        """Desenha o ponto ativo com arco de progresso."""
        x, y   = pt
        prog   = n_samples / CALIB_SAMPLES
        radius = 28

        # Anel base
        cv2.circle(canvas, (x, y), radius, (180, 180, 180), 2)

        # Arco de progresso (aproximado com pontos)
        angle = int(360 * prog)
        for a in range(0, angle, 3):
            r = np.radians(a - 90)
            px = int(x + (radius - 3) * np.cos(r))
            py = int(y + (radius - 3) * np.sin(r))
            g  = int(80 + 175 * prog)
            cv2.circle(canvas, (px, py), 2, (0, g, 100), -1)

        # Centro
        cv2.circle(canvas, (x, y), 8, (0, 220, 120), -1)

        # Porcentagem
        pct = f'{int(prog * 100)}%'
        cv2.putText(canvas, pct, (x - 18, y + radius + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
