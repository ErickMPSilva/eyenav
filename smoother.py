"""
Suavizador de sinal 2D por Média Móvel Exponencial (EMA).

A EMA é amplamente usada em sistemas de eye tracking de baixo custo
por equilibrar responsividade e estabilidade com custo computacional O(1).

Equação:
  s[t] = α · x[t] + (1 − α) · s[t−1]

α = 0.12 (padrão):
  - Suaviza tremulações de alta frequência da câmera (jitter)
  - Mantém latência perceptível < 200 ms a 30 fps
  - Referência: Munn et al. (2008) recomendam α ∈ [0.1, 0.2]
"""

from config import SMOOTH_ALPHA


class EMASmoother:
    """Suavizador EMA para coordenadas 2D do ponto de olhar."""

    def __init__(self, alpha: float = SMOOTH_ALPHA):
        self.alpha = alpha
        self._sx: float | None = None
        self._sy: float | None = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        """
        Atualiza com nova leitura e retorna valor suavizado.

        Na primeira chamada retorna a própria leitura (sem histórico).
        """
        if self._sx is None:
            self._sx = float(x)
            self._sy = float(y)
        else:
            self._sx = self.alpha * x + (1 - self.alpha) * self._sx
            self._sy = self.alpha * y + (1 - self.alpha) * self._sy
        return self._sx, self._sy

    def reset(self):
        """Limpa o histórico (usar após recalibração)."""
        self._sx = None
        self._sy = None

    @property
    def value(self) -> tuple[float, float]:
        return (self._sx or 0.0, self._sy or 0.0)
