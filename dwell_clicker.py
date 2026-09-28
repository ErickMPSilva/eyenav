"""
Seleção por tempo de fixação (Dwell Click).

O dwell click é o principal mecanismo de interação do EyeNav.
Permite ao usuário "clicar" mantendo o olhar fixo sobre um elemento
da tela por DWELL_TIME_MS milissegundos — sem qualquer hardware adicional.

Algoritmo:
  1. Recebe posição do olhar (já suavizada) a cada frame.
  2. Define um ponto de âncora na primeira leitura.
  3. Mede distância euclidiana do olhar atual até a âncora.
  4. Se distância > DWELL_RADIUS_PX → reinicia âncora e temporizador.
  5. Se distância ≤ DWELL_RADIUS_PX e tempo ≥ DWELL_TIME_MS → clique.
  6. Após clicar, marca como 'clicado' para evitar disparo duplo.
  7. Só rearma quando o olhar se afasta mais que DWELL_REARM_PX
     (maior que o raio), para que tremores não gerem cliques repetidos.

A execução do clique é delegada ao mouse_backend.

Referências:
  Sibert & Jacob (2000) — primeira avaliação sistemática de dwell time
  como mecanismo de seleção em interfaces gráficas (ACM CHI).
  Majaranta & Bulling (2014) — revisão de técnicas de seleção ocular.
"""

import time

import mouse_backend
from config import DWELL_TIME_MS, DWELL_RADIUS_PX, DWELL_REARM_PX


class DwellClicker:
    """Gerencia a lógica de clique por fixação do olhar."""

    def __init__(self):
        self._ax:      float | None = None
        self._ay:      float | None = None
        self._t_start: float | None = None
        self._fired:   bool         = False
        self._click_count: int      = 0     # para debug

    def update(self, gx: int, gy: int) -> bool:
        """
        Atualiza estado com a posição atual do olhar.

        Returns True se um clique foi disparado neste frame.
        """
        if self._ax is None:
            self._reset_anchor(gx, gy)
            return False

        dist = _dist(gx, gy, self._ax, self._ay)

        if self._fired:
            # Após um clique, só rearma se o olhar realmente sair do alvo.
            # Evita que tremores ou piscadas gerem cliques repetidos.
            if dist > DWELL_REARM_PX:
                self._reset_anchor(gx, gy)
            return False

        if dist > DWELL_RADIUS_PX:
            self._reset_anchor(gx, gy)
            return False

        elapsed_ms = (time.time() - self._t_start) * 1000
        if elapsed_ms >= DWELL_TIME_MS:
            ax, ay = int(self._ax), int(self._ay)
            self._click_count += 1
            print(f'[DwellClicker] CLIQUE #{self._click_count} '
                  f'em ({ax}, {ay}) — backend: {mouse_backend.BACKEND}')
            mouse_backend.click(ax, ay)
            self._fired = True
            return True

        return False

    @property
    def progress(self) -> float:
        """Progresso do dwell atual [0.0, 1.0]."""
        if self._t_start is None or self._fired:
            return 0.0
        elapsed = (time.time() - self._t_start) * 1000
        return min(elapsed / DWELL_TIME_MS, 1.0)

    @property
    def anchor(self) -> tuple:
        return (self._ax, self._ay)

    def reset(self):
        self._ax      = None
        self._ay      = None
        self._t_start = None
        self._fired   = False

    def _reset_anchor(self, gx: int, gy: int):
        self._ax      = float(gx)
        self._ay      = float(gy)
        self._t_start = time.time()
        self._fired   = False


def _dist(x1, y1, x2, y2) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5