"""
Rola a página automaticamente quando o olhar está na borda
superior ou inferior da tela.

  Olhar no topo (< SCROLL_ZONE_PX)                → scroll para cima
  Olhar no baixo (> screen_h - SCROLL_ZONE_PX)    → scroll para baixo

O scroll é contínuo enquanto o olhar permanecer na zona,
com intervalo de SCROLL_INTERVAL_MS entre ticks para não
ser agressivo demais. A execução é delegada ao mouse_backend.
"""

import time

import mouse_backend
from config import SCROLL_ZONE_PX, SCROLL_SPEED, SCROLL_INTERVAL_MS


class EdgeScroller:
    """
    Monitora a posição vertical do olhar e dispara scroll nas bordas.

    Uso:
        scroller = EdgeScroller(screen_h=1080)
        # No loop principal, após obter gaze_sy:
        scroller.update(gaze_sy)
    """

    def __init__(self, screen_h: int):
        self.screen_h    = screen_h
        self._last_tick  = 0.0      # timestamp do último scroll disparado
        self._zone       = None     # 'top' | 'bottom' | None
        self._zone_entry = 0.0      # quando entrou na zona (pequeno delay inicial)

    def update(self, gaze_y: int) -> str | None:
        """
        Verifica se o olhar está na zona de scroll e dispara se necessário.

        Returns
        -------
        'up' | 'down' | None
        """
        now = time.time()

        # Determina em qual zona o olhar está
        if gaze_y < SCROLL_ZONE_PX:
            new_zone = 'top'
        elif gaze_y > self.screen_h - SCROLL_ZONE_PX:
            new_zone = 'bottom'
        else:
            new_zone = None

        # Entrou em nova zona → registra o momento
        if new_zone != self._zone:
            self._zone       = new_zone
            self._zone_entry = now
            return None

        # Fora de qualquer zona
        if self._zone is None:
            return None

        # Exige 300ms na zona antes de começar a scrollar (evita scroll acidental)
        if now - self._zone_entry < 0.3:
            return None

        # Respeita o intervalo entre ticks
        if now - self._last_tick < SCROLL_INTERVAL_MS / 1000:
            return None

        # Dispara scroll
        direction = 1 if self._zone == 'top' else -1
        mouse_backend.scroll(direction * SCROLL_SPEED)
        self._last_tick = now
        return 'up' if direction > 0 else 'down'

    @property
    def active_zone(self) -> str | None:
        """Retorna a zona ativa ('top', 'bottom' ou None) para o debug."""
        return self._zone

    def reset(self):
        self._zone       = None
        self._last_tick  = 0.0
        self._zone_entry = 0.0