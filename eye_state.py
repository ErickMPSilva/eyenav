"""
Detector de olho fechado para bloquear o olhar durante piscadas.

Problema:
  Em toda piscada — inclusive as naturais, a cada 3–5 s — a pálpebra
  cobre a íris e o MediaPipe desloca os landmarks. O ponto de olhar
  estimado "salta" dezenas de pixels e volta. Esse salto:
    - faz o cursor tremer a cada piscada;
    - sai do raio do dwell e REARMA o dwell click no mesmo lugar,
      que então dispara ~1,5 s depois da piscada.

Solução:
  Enquanto o olho está fechando/fechado, e por BLINK_GRACE_MS após
  reabrir, o olhar é ignorado: cursor, dwell e scroll ficam congelados.

  O limiar é adaptativo: acompanha o EAR de olho aberto por média móvel
  lenta e considera "piscando" quando EAR < BLINK_SUPPRESS_RATIO × EAR_aberto.
  O limiar é mais sensível que o do BlinkClicker, para capturar a
  pálpebra já no início do fechamento.

Funciona sempre, com o blink click ligado ou desligado.
"""

import time

from config import (BLINK_EAR_THRESHOLD, BLINK_SUPPRESS_RATIO,
                    BLINK_GRACE_MS)

_BASELINE_RATE = 0.02


class EyeState:
    """Indica se o olhar do frame atual deve ser ignorado."""

    def __init__(self):
        self.reset()

    def update(self, ear: float) -> bool:
        """
        Atualiza com o EAR do frame.

        Returns True se o olhar deve ser ignorado neste frame.
        """
        now = time.time()

        if ear < self.threshold:
            self._closed = True
        else:
            if self._closed:
                self._closed   = False
                self._t_reopen = now
                self.blinks   += 1
            # Só aprende o EAR de olho aberto fora do período de graça
            if (now - self._t_reopen) * 1000 >= BLINK_GRACE_MS:
                if self._baseline is None:
                    self._baseline = ear
                else:
                    self._baseline += _BASELINE_RATE * (ear - self._baseline)

        self.blocked = (self._closed or
                        (now - self._t_reopen) * 1000 < BLINK_GRACE_MS)
        return self.blocked

    @property
    def threshold(self) -> float:
        if self._baseline is None:
            return BLINK_EAR_THRESHOLD
        return self._baseline * BLINK_SUPPRESS_RATIO

    def reset(self):
        self._closed   = False
        self._t_reopen = 0.0
        self._baseline = None
        self.blocked   = False
        self.blinks    = 0