"""
Rola a página automaticamente quando o olhar está na borda
superior ou inferior da tela.

  Olhar no topo (< SCROLL_ZONE_PX)                → scroll para cima
  Olhar no baixo (> screen_h - SCROLL_ZONE_PX)    → scroll para baixo

O scroll é contínuo enquanto o olhar permanecer na zona,
com intervalo de SCROLL_INTERVAL_MS entre ticks para não
ser agressivo demais.
"""

import os
import time
import subprocess
import threading

from config import SCROLL_ZONE_PX, SCROLL_SPEED, SCROLL_INTERVAL_MS

# Backend
_DISPLAY = os.environ.get('DISPLAY', ':0')
_WAYLAND = bool(os.environ.get('WAYLAND_DISPLAY'))

def _cmd_ok(cmd):
    try:
        return subprocess.run(['which', cmd], capture_output=True, timeout=2).returncode == 0
    except Exception:
        return False

_HAS_XDOTOOL = _cmd_ok('xdotool')
_HAS_YDOTOOL = _cmd_ok('ydotool')

try:
    from pynput.mouse import Controller as _MC
    _pm = _MC()
    _HAS_PYNPUT = True
except Exception:
    _pm = None
    _HAS_PYNPUT = False

if _WAYLAND and _HAS_YDOTOOL:
    _BACKEND = 'ydotool'
elif _HAS_XDOTOOL:
    _BACKEND = 'xdotool'
elif _HAS_PYNPUT:
    _BACKEND = 'pynput'
else:
    _BACKEND = 'none'


def _do_scroll(direction: int):
    """
    Executa o scroll.
    direction: +1 = cima, -1 = baixo
    """
    def _t():
        try:
            if _BACKEND == 'xdotool':
                btn = '4' if direction > 0 else '5'
                for _ in range(SCROLL_SPEED):
                    subprocess.run(
                        ['xdotool', 'click', btn],
                        capture_output=True, timeout=1,
                        env={**os.environ, 'DISPLAY': _DISPLAY}
                    )
            elif _BACKEND == 'ydotool':
                # ydotool scroll: button 4 = cima, 5 = baixo
                btn = '0x40004' if direction > 0 else '0x40005'
                for _ in range(SCROLL_SPEED):
                    subprocess.run(['ydotool', 'click', btn],
                                   capture_output=True, timeout=1)
            elif _BACKEND == 'pynput' and _pm:
                _pm.scroll(0, direction * SCROLL_SPEED)
        except Exception:
            pass

    threading.Thread(target=_t, daemon=True).start()


# Classe principal

class EdgeScroller:
    """
    Monitora a posição vertical do olhar e dispara scroll nas bordas.

    Uso:
        scroller = EdgeScroller(screen_h=1080)
        # No loop principal, após obter gaze_sy:
        scroller.update(gaze_sy)
    """

    def __init__(self, screen_h: int):
        self.screen_h   = screen_h
        self._last_tick = 0.0       # timestamp do último scroll disparado
        self._zone      = None      # 'top' | 'bottom' | None
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
        _do_scroll(direction)
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