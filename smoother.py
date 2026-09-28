"""
Suavização 2D do ponto de olhar.

O sinal de olhar de uma webcam é ruidoso: mesmo com o olho parado,
a posição estimada da íris oscila alguns pixels a cada frame, e o
modelo de calibração amplifica essa oscilação até dezenas de pixels
na tela. Dois estágios tratam o problema:

1. Filtro de mediana (janela MEDIAN_WINDOW)
   Remove picos isolados (landmark que "pula" em um frame, início
   de piscada) sem atrasar mudanças reais de direção do olhar.

2. Filtro One Euro (Casiez, Roussel & Vogel, 2012)
   Passa-baixa com frequência de corte adaptativa à velocidade:
     - olhar parado  → corte baixo  → muita suavização (sem tremor)
     - olhar movendo → corte alto   → pouca suavização (sem atraso)
   Resolve o dilema da EMA fixa, em que reduzir o tremor
   inevitavelmente aumenta o atraso.

     α(fc)   = 1 / (1 + τ/Δt),   τ = 1 / (2π·fc)
     v̂[t]    = passa-baixa(velocidade, d_cutoff)
     fc      = min_cutoff + β·|v̂[t]|
     ŝ[t]    = α(fc)·x[t] + (1 − α(fc))·ŝ[t−1]

A EMA simples (Munn et al., 2008) continua disponível para
comparação no Capítulo 4 (SMOOTHER = 'ema').

Referência:
  CASIEZ, G.; ROUSSEL, N.; VOGEL, D. 1€ Filter: a simple speed-based
  low-pass filter for noisy input in interactive systems. ACM CHI, 2012.
"""

import math
import time
from collections import deque

import numpy as np

from config import (SMOOTHER, SMOOTH_ALPHA, MEDIAN_WINDOW,
                    ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)


# EMA simples (referência)

class EMASmoother:
    """Suavizador EMA com α fixo."""

    def __init__(self, alpha: float = SMOOTH_ALPHA):
        self.alpha = alpha
        self._sx: float | None = None
        self._sy: float | None = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        if self._sx is None:
            self._sx, self._sy = float(x), float(y)
        else:
            self._sx = self.alpha * x + (1 - self.alpha) * self._sx
            self._sy = self.alpha * y + (1 - self.alpha) * self._sy
        return self._sx, self._sy

    def reset(self):
        self._sx = self._sy = None


# One Euro

class OneEuroSmoother:
    """Filtro One Euro 2D (corte adaptativo à velocidade do olhar)."""

    def __init__(self, min_cutoff: float = ONE_EURO_MIN_CUTOFF,
                 beta: float = ONE_EURO_BETA,
                 d_cutoff: float = ONE_EURO_D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self.reset()

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, x: float, y: float,
               t: float | None = None) -> tuple[float, float]:
        """t: instante da amostra (s). None = relógio atual.
        Informar t permite reprocessar sequências gravadas (fine tuning)."""
        now = time.time() if t is None else t
        p = np.array([x, y], dtype=float)

        if self._p is None:
            self._p, self._v, self._t = p, np.zeros(2), now
            return float(p[0]), float(p[1])

        dt = max(now - self._t, 1e-3)
        self._t = now

        # Velocidade suavizada (px/s)
        v_raw   = (p - self._p) / dt
        a_d     = self._alpha(self.d_cutoff, dt)
        self._v = a_d * v_raw + (1 - a_d) * self._v

        # Corte adaptativo: único para os dois eixos (movimento do olhar é 2D)
        speed  = float(np.hypot(*self._v))
        cutoff = self.min_cutoff + self.beta * speed
        a      = self._alpha(cutoff, dt)

        self._p = a * p + (1 - a) * self._p
        return float(self._p[0]), float(self._p[1])

    def reset(self):
        self._p = None
        self._v = None
        self._t = None


# Pipeline: mediana + suavizador

class GazeSmoother:
    """Mediana deslizante seguida do suavizador escolhido em SMOOTHER."""

    def __init__(self):
        self._buf = deque(maxlen=max(1, MEDIAN_WINDOW))
        self._filt = OneEuroSmoother() if SMOOTHER == 'one_euro' else EMASmoother()
        print(f'[Smoother] mediana({MEDIAN_WINDOW}) + '
              f'{"One Euro" if SMOOTHER == "one_euro" else "EMA"}')

    def update(self, x: float, y: float) -> tuple[float, float]:
        self._buf.append((x, y))
        mx, my = np.median(np.array(self._buf), axis=0)
        return self._filt.update(mx, my)

    def reset(self):
        self._buf.clear()
        self._filt.reset()

    def set_min_cutoff(self, value: float):
        """Ajusta o corte mínimo do One Euro (definido pelo fine tuning)."""
        if isinstance(self._filt, OneEuroSmoother):
            self._filt.min_cutoff = value
            print(f'[Smoother] min_cutoff ajustado para {value:.3f} Hz')