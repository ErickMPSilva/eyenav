"""
Dispara clique simples, duplo e direito por piscada intencional.

Tipos de piscada:
  Fechamento  (BLINK_MIN_MS  → BLINK_MAX_MS)           → clique simples (esquerdo)
  Dois fechamentos (pausa ≤ DOUBLE_INTERVAL)           → duplo clique
  Fechamento longo (BLINK_RIGHT_MIN_MS → RIGHT_MAX_MS) → clique direito

Qualquer fechamento menor que BLINK_MIN_MS (padrão 2 s) é ignorado.
Piscadas naturais duram ~100–400 ms, portanto nunca disparam clique.

Limiar adaptativo:
  O EAR de olho aberto varia entre pessoas, iluminação e direção do
  olhar (olhar para baixo abaixa a pálpebra). Por isso o sistema
  acompanha o EAR de olho aberto por média móvel lenta e considera o
  olho fechado quando EAR < BLINK_EAR_RATIO × EAR_aberto.
  Até ter referência, usa BLINK_EAR_THRESHOLD.

Posição do clique:
  Durante a piscada a pálpebra desloca os landmarks da íris e o ponto
  de olhar estimado "salta". O clique usa a posição registrada no
  instante em que o olho fechou, não a da reabertura.

A execução do clique é delegada ao mouse_backend.

Referência: Soukupová & Čech (2016) — Real-Time Eye Blink Detection.
"""

import time

import mouse_backend
from config import (BLINK_EAR_THRESHOLD, BLINK_EAR_RATIO,
                    BLINK_MIN_MS, BLINK_MAX_MS,
                    BLINK_RIGHT_MIN_MS, BLINK_RIGHT_MAX_MS,
                    BLINK_DOUBLE_INTERVAL_MS)

_BASELINE_RATE = 0.02     # velocidade de adaptação do EAR de olho aberto
_COOLDOWN_S    = 0.3      # pausa pós-clique


def _fire(action, x: int, y: int, label: str, duracao_ms: float, thr: float):
    """Dispara a ação de clique e registra no console."""
    action(x, y)
    print(f'[BlinkClicker] {label} → ({x}, {y})  [{mouse_backend.BACKEND}] '
          f'| limiar={thr:.3f} | duração={duracao_ms:.0f}ms')


class BlinkClicker:
    """Detecta piscadas intencionais e dispara o tipo de clique correto."""

    def __init__(self):
        self.reset()
        self._n = 0

    # API pública

    def update(self, ear: float, gaze_x: int, gaze_y: int) -> str | None:
        """
        Processa o EAR do frame e dispara o clique adequado se detectado.

        Returns
        -------
        'single' | 'double' | 'right' | None
        """
        now = time.time()
        thr = self.threshold

        if not self._eye_closed:
            if ear < thr:
                # Olho acabou de fechar → congela a posição do clique
                self._eye_closed = True
                self._t_fechou   = now
                self._click_x    = gaze_x
                self._click_y    = gaze_y
            else:
                # Olho aberto → atualiza a referência de EAR
                if self._baseline is None:
                    self._baseline = ear
                else:
                    self._baseline += _BASELINE_RATE * (ear - self._baseline)
            return None

        # Olho fechado → aguarda reabrir
        if ear < thr:
            return None

        self._eye_closed = False
        duracao_ms = (now - self._t_fechou) * 1000

        if now - self._cooldown < _COOLDOWN_S:
            return None
        if duracao_ms < BLINK_MIN_MS:
            return None            # piscada involuntária
        if duracao_ms > BLINK_RIGHT_MAX_MS:
            return None            # olho fechado tempo demais

        self._n += 1
        x, y = self._click_x, self._click_y

        if duracao_ms <= BLINK_MAX_MS:
            # Pausa entre a reabertura anterior e este fechamento
            pausa_ms = (self._t_fechou - self._t_last_click) * 1000
            if self._t_last_click > 0 and pausa_ms <= BLINK_DOUBLE_INTERVAL_MS:
                _fire(mouse_backend.double_click, x, y,
                      f'DUPLO CLIQUE #{self._n}', duracao_ms, thr)
                self._t_last_click = 0.0
                self._cooldown     = now
                return 'double'

            _fire(mouse_backend.click, x, y,
                  f'CLIQUE SIMPLES #{self._n}', duracao_ms, thr)
            self._t_last_click = now
            self._cooldown     = now
            return 'single'

        # Piscada longa → clique direito
        _fire(mouse_backend.right_click, x, y,
              f'CLIQUE DIREITO #{self._n}', duracao_ms, thr)
        self._t_last_click = 0.0
        self._cooldown     = now
        return 'right'

    @property
    def threshold(self) -> float:
        """Limiar de EAR atual (adaptativo)."""
        if self._baseline is None:
            return BLINK_EAR_THRESHOLD
        return self._baseline * BLINK_EAR_RATIO

    @property
    def baseline(self) -> float | None:
        """EAR médio de olho aberto aprendido."""
        return self._baseline

    @property
    def eye_closed(self) -> bool:
        return self._eye_closed

    def reset(self):
        self._eye_closed   = False
        self._t_fechou     = 0.0
        self._t_last_click = 0.0
        self._cooldown     = 0.0
        self._click_x      = 0
        self._click_y      = 0
        self._baseline     = None