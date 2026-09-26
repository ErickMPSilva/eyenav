"""
Controle do cursor do sistema operacional via pyautogui.

Restringe o movimento às bordas da tela e expõe ações de
navegação (scroll) para uso com eye tracking.
"""

import os
import subprocess
from config import CURSOR_MOVE

_DISPLAY = os.environ.get('DISPLAY', ':0')
_WAYLAND = bool(os.environ.get('WAYLAND_DISPLAY'))

def _cmd_ok(cmd):
    try:
        return subprocess.run(['which', cmd], capture_output=True,
                              timeout=2).returncode == 0
    except Exception:
        return False

_HAS_XDOTOOL = _cmd_ok('xdotool')
_HAS_YDOTOOL = _cmd_ok('ydotool')

try:
    from pynput.mouse import Controller as _MC, Button as _Btn
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

print(f'[CursorController] Backend: {_BACKEND}')


class CursorController:
    """Move o cursor do SO para o ponto de olhar estimado."""

    def __init__(self, screen_w: int, screen_h: int):
        self.sw = screen_w
        self.sh = screen_h

    def move(self, x: float, y: float):
        if not CURSOR_MOVE:
            return
        cx = int(max(0, min(x, self.sw - 1)))
        cy = int(max(0, min(y, self.sh - 1)))

        try:
            if _BACKEND == 'xdotool':
                subprocess.Popen(
                    ['xdotool', 'mousemove', str(cx), str(cy)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    env={**os.environ, 'DISPLAY': _DISPLAY}
                )
            elif _BACKEND == 'ydotool':
                subprocess.Popen(
                    ['ydotool', 'mousemove', '--absolute',
                     '-x', str(cx), '-y', str(cy)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            elif _BACKEND == 'pynput' and _pm:
                _pm.position = (cx, cy)
        except Exception:
            pass

    def scroll(self, clicks: int):
        try:
            if _BACKEND == 'xdotool':
                btn = '4' if clicks > 0 else '5'
                for _ in range(abs(clicks)):
                    subprocess.Popen(
                        ['xdotool', 'click', btn],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        env={**os.environ, 'DISPLAY': _DISPLAY}
                    )
            elif _BACKEND == 'pynput' and _pm:
                _pm.scroll(0, clicks)
        except Exception:
            pass