"""
Tela de calibração visual em janela OpenCV fullscreen.

Fluxo (inspirado na rotina de grade densa do EyeTrax — Zhang, 2025):

  1. Espera de rosto
     A calibração só começa depois que o rosto é detectado, com o olho
     aberto, de forma contínua por CALIB_FACE_WAIT_MS. Nesse período
     também é medido o EAR de referência (olho aberto) do usuário.

  2. Para cada ponto da grade (percurso em serpentina):
     a) Pulso (CALIB_PULSE_MS): o círculo pulsa para atrair o olhar.
        NADA é coletado — o olho ainda está em trânsito (sacada).
     b) Captura (CALIB_CAPTURE_MS): um arco se fecha em volta do ponto.
        CADA frame válido vira uma amostra de treino separada.
        Frames com piscada ou sem rosto são descartados.

Por que guardar cada frame, e não a média por ponto?
  Com a média, o modelo treina com apenas ROWS×COLS exemplos (ex.: 16)
  e nunca "vê" o ruído real do sinal; os coeficientes ficam instáveis
  e amplificam o tremor. Com cada frame, são centenas de exemplos
  (≈ 30 fps × 1 s × 16 pontos ≈ 400–500). A regressão Ridge passa a
  ajustar a tendência média sob o ruído real, o que estabiliza o
  mapeamento.

Percurso em serpentina: a linha seguinte começa pela coluna onde a
anterior terminou, evitando saltos longos atravessando a tela.

Tecla ESC durante a calibração cancela a operação.
"""

import time

import cv2
import numpy as np
from config import (CALIB_ROWS, CALIB_COLS, CALIB_MARGIN,
                    CALIB_FACE_WAIT_MS, CALIB_PULSE_MS, CALIB_CAPTURE_MS,
                    BLINK_SUPPRESS_RATIO)

_GREEN = (0, 255, 0)
_WHITE = (255, 255, 255)
_RED   = (0, 0, 255)
_GRAY  = (90, 90, 90)
_MIN_SAMPLES_PER_POINT = 10


# Helpers

def build_calibration_points(sw: int, sh: int) -> list[tuple[int, int]]:
    """Grade ROWS × COLS em serpentina, com margem proporcional à tela."""
    mx, my = int(sw * CALIB_MARGIN), int(sh * CALIB_MARGIN)
    gw, gh = sw - 2 * mx, sh - 2 * my
    step_x = gw / (CALIB_COLS - 1) if CALIB_COLS > 1 else 0
    step_y = gh / (CALIB_ROWS - 1) if CALIB_ROWS > 1 else 0

    pts = []
    for r in range(CALIB_ROWS):
        cols = range(CALIB_COLS) if r % 2 == 0 else range(CALIB_COLS - 1, -1, -1)
        for c in cols:
            pts.append((mx + int(c * step_x), my + int(r * step_y)))
    return pts


def _smoothstep(t: float) -> float:
    """Suavização da animação: começa e termina devagar."""
    return t * t * (3 - 2 * t)


# Classe principal

class CalibrationScreen:
    """Janela de calibração visual."""

    WIN = 'EyeNav - Calibracao'

    def __init__(self, sw: int, sh: int):
        self.sw      = sw
        self.sh      = sh
        self.points  = build_calibration_points(sw, sh)
        self._ear_ref: float | None = None

    def run(self, estimator, cap) -> tuple[list, list]:
        """
        Executa a calibração completa.

        Returns
        -------
        features : list of (gaze_x, gaze_y)   — uma por frame válido
        targets  : list of (screen_x, screen_y)
            Listas vazias se o usuário cancelar com ESC.
        """
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.WIN, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        if not self._wait_for_face(estimator, cap):
            cv2.destroyWindow(self.WIN)
            return [], []

        features: list = []
        targets:  list = []

        for idx, pt in enumerate(self.points):
            if not self._pulse(cap, pt, idx):
                cv2.destroyWindow(self.WIN)
                return [], []

            samples = self._capture(estimator, cap, pt, idx)
            if samples is None:
                cv2.destroyWindow(self.WIN)
                return [], []

            if len(samples) < _MIN_SAMPLES_PER_POINT:
                print(f'[Calibração] AVISO: ponto {idx + 1} {pt} com apenas '
                      f'{len(samples)} amostras (piscadas ou rosto perdido).')

            features.extend(samples)
            targets.extend([pt] * len(samples))

        self._show_message('Calibracao concluida!', (0, 255, 100), 1000)
        cv2.destroyWindow(self.WIN)

        print(f'[Calibração] {len(features)} amostras em '
              f'{len(self.points)} pontos ({CALIB_ROWS}×{CALIB_COLS}).')
        return features, targets

    # Etapas

    def _wait_for_face(self, estimator, cap) -> bool:
        """
        Aguarda rosto detectado com olho aberto por CALIB_FACE_WAIT_MS
        contínuos, mostrando um disco em contagem regressiva.
        Mede o EAR de referência (mediana) nesse intervalo.
        """
        t_start = None
        ears: list[float] = []

        while True:
            frame = self._read(cap)
            if frame is None:
                continue
            feat = estimator.process(frame)
            canvas = self._blank()
            now = time.time()

            if feat is not None and self._eye_open(feat['ear'], ears):
                if t_start is None:
                    t_start = now
                    ears.clear()
                ears.append(feat['ear'])

                t = (now - t_start) * 1000 / CALIB_FACE_WAIT_MS
                if t >= 1.0:
                    self._ear_ref = float(np.median(ears))
                    return True
                ang = 360 * (1 - _smoothstep(t))
                cv2.ellipse(canvas, (self.sw // 2, self.sh // 2), (50, 50),
                            0, -90, -90 + ang, _GREEN, -1)
            else:
                t_start = None
                self._center_text(canvas, 'Rosto nao detectado', _RED)

            self._hint(canvas)
            cv2.imshow(self.WIN, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                return False

    def _pulse(self, cap, pt: tuple, idx: int) -> bool:
        """Círculo pulsante: atrai o olhar. Não coleta amostras."""
        t0 = time.time()
        while (elapsed := time.time() - t0) * 1000 < CALIB_PULSE_MS:
            self._read(cap)          # mantém o buffer da câmera atualizado
            canvas = self._blank()
            radius = 15 + int(15 * abs(np.sin(2 * np.pi * elapsed)))
            cv2.circle(canvas, pt, radius, _GREEN, -1)
            self._progress(canvas, idx)
            cv2.imshow(self.WIN, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                return False
        return True

    def _capture(self, estimator, cap, pt: tuple, idx: int) -> list | None:
        """Coleta uma amostra por frame válido enquanto o arco se fecha."""
        samples: list = []
        t0 = time.time()
        while (elapsed := time.time() - t0) * 1000 < CALIB_CAPTURE_MS:
            frame = self._read(cap)
            if frame is None:
                continue

            canvas = self._blank()
            cv2.circle(canvas, pt, 20, _GREEN, -1)
            t   = elapsed * 1000 / CALIB_CAPTURE_MS
            ang = 360 * (1 - _smoothstep(t))
            cv2.ellipse(canvas, pt, (40, 40), 0, -90, -90 + ang, _WHITE, 4)
            self._progress(canvas, idx)
            cv2.imshow(self.WIN, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                return None

            feat = estimator.process(frame)
            if feat is not None and self._eye_open(feat['ear']):
                samples.append(feat['features'])
        return samples

    # Utilidades

    def _eye_open(self, ear: float, recent: list | None = None) -> bool:
        """
        Olho aberto = EAR ≥ BLINK_SUPPRESS_RATIO × referência.
        Antes de haver referência, compara com a mediana recente.
        """
        ref = self._ear_ref
        if ref is None:
            if not recent:
                return True
            ref = float(np.median(recent))
        return ear >= ref * BLINK_SUPPRESS_RATIO

    @staticmethod
    def _read(cap):
        ok, frame = cap.read()
        return cv2.flip(frame, 1) if ok else None

    def _blank(self) -> np.ndarray:
        return np.zeros((self.sh, self.sw, 3), dtype=np.uint8)

    def _center_text(self, canvas, txt, color):
        size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(canvas, txt,
                    ((self.sw - size[0]) // 2, (self.sh + size[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    def _progress(self, canvas, idx: int):
        cv2.putText(canvas, f'{idx + 1}/{len(self.points)}',
                    (20, self.sh - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, _GRAY, 1)
        self._hint(canvas)

    def _hint(self, canvas):
        cv2.putText(canvas, 'ESC = cancelar', (self.sw - 150, self.sh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GRAY, 1)

    def _show_message(self, txt, color, ms):
        canvas = self._blank()
        self._center_text(canvas, txt, color)
        cv2.imshow(self.WIN, canvas)
        cv2.waitKey(ms)