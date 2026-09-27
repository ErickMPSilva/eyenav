"""
Dispara clique simples, duplo e direito por piscada intencional.

Tipos de piscada:
  Curta  (BLINK_MIN_MS  → BLINK_MAX_MS)       → clique simples (esquerdo)
  Dupla  (segunda piscada em DOUBLE_INTERVAL)  → duplo clique
  Longa  (BLINK_RIGHT_MIN_MS → BLINK_RIGHT_MAX_MS) → clique direito

Piscadas involuntárias (< BLINK_MIN_MS) são ignoradas silenciosamente.

Referência: Soukupová & Čech (2016) — Real-Time Eye Blink Detection.
"""

import time
import os
import subprocess
import threading

from config import (BLINK_EAR_THRESHOLD,
                    BLINK_MIN_MS, BLINK_MAX_MS,
                    BLINK_RIGHT_MIN_MS, BLINK_RIGHT_MAX_MS,
                    BLINK_DOUBLE_INTERVAL_MS)

# Backend de clique (mesmo do dwell_clicker)
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


def _run(cmd):
    try:
        subprocess.run(cmd, timeout=1, capture_output=True,
                       env={**os.environ, 'DISPLAY': _DISPLAY})
    except Exception:
        pass


def _click_single(x, y):
    if _BACKEND == 'xdotool':
        _run(['xdotool', 'mousemove', '--sync', str(x), str(y)])
        _run(['xdotool', 'click', '--clearmodifiers', '1'])
    elif _BACKEND == 'ydotool':
        _run(['ydotool', 'mousemove', '--absolute', '-x', str(x), '-y', str(y)])
        time.sleep(0.05)
        _run(['ydotool', 'click', '0x40001'])
    elif _BACKEND == 'pynput' and _pm:
        _pm.position = (x, y)
        time.sleep(0.02)
        _pm.click(_Btn.left)


def _click_double(x, y):
    if _BACKEND == 'xdotool':
        _run(['xdotool', 'mousemove', '--sync', str(x), str(y)])
        _run(['xdotool', 'click', '--clearmodifiers', '--repeat', '2', '--delay', '80', '1'])
    elif _BACKEND == 'ydotool':
        _run(['ydotool', 'mousemove', '--absolute', '-x', str(x), '-y', str(y)])
        time.sleep(0.05)
        _run(['ydotool', 'click', '0x40001'])
        time.sleep(0.08)
        _run(['ydotool', 'click', '0x40001'])
    elif _BACKEND == 'pynput' and _pm:
        _pm.position = (x, y)
        time.sleep(0.02)
        _pm.click(_Btn.left, 2)


def _click_right(x, y):
    if _BACKEND == 'xdotool':
        _run(['xdotool', 'mousemove', '--sync', str(x), str(y)])
        _run(['xdotool', 'click', '--clearmodifiers', '3'])
    elif _BACKEND == 'ydotool':
        _run(['ydotool', 'mousemove', '--absolute', '-x', str(x), '-y', str(y)])
        time.sleep(0.05)
        _run(['ydotool', 'click', '0x40004'])   # right button
    elif _BACKEND == 'pynput' and _pm:
        _pm.position = (x, y)
        time.sleep(0.02)
        _pm.click(_Btn.right)


def _fire(fn, x, y, label):
    def _t():
        fn(x, y)
    threading.Thread(target=_t, daemon=True).start()
    print(f'[BlinkClicker] {label} → ({x}, {y})  [{_BACKEND}]')


# Classe principal

class BlinkClicker:
    """
    Detecta piscadas intencionais e dispara o tipo de clique correto.

    Estados internos:
      eye_closed     : olho está fechado agora?
      t_fechou       : quando o olho fechou
      t_ultimo_click : quando foi o último clique simples (para detectar duplo)
      n              : contador de cliques disparados
    """

    def __init__(self):
        self._eye_closed   = False
        self._t_fechou     = 0.0
        self._t_last_click = 0.0    # timestamp do último clique simples
        self._last_x       = 0
        self._last_y       = 0
        self._cooldown     = 0.0
        self._n            = 0

    def update(self, ear: float, gaze_x: int, gaze_y: int) -> str | None:
        """
        Processa o EAR do frame e dispara o clique adequado se detectado.

        Returns
        -------
        'single' | 'double' | 'right' | None
        """
        now = time.time()

        # Cooldown pós-clique (evita disparos em cascata)
        if now - self._cooldown < 0.3:
            return None

        if not self._eye_closed:
            # Olho aberto → verifica se fechou
            if ear < BLINK_EAR_THRESHOLD:
                self._eye_closed = True
                self._t_fechou   = now
        else:
            # Olho fechado → verifica se abriu
            if ear >= BLINK_EAR_THRESHOLD:
                duracao_ms = (now - self._t_fechou) * 1000
                self._eye_closed = False
                self._n += 1

                if duracao_ms < BLINK_MIN_MS:
                    # Piscada involuntária — ignora
                    return None

                elif duracao_ms <= BLINK_MAX_MS:
                    # Piscada curta → simples ou duplo
                    intervalo_ms = (now - self._t_last_click) * 1000
                    if intervalo_ms <= BLINK_DOUBLE_INTERVAL_MS and self._t_last_click > 0:
                        # Segunda piscada dentro da janela → duplo clique
                        _fire(_click_double, gaze_x, gaze_y,
                              f'DUPLO CLIQUE #{self._n} ({duracao_ms:.0f} ms)')
                        print(f'[BlinkClicker] EAR threshold={BLINK_EAR_THRESHOLD} '
                            f'| duração={duracao_ms:.0f}ms')
                        self._t_last_click = 0.0     # reseta janela
                        self._cooldown = now
                        return 'double'
                    else:
                        # Primeira piscada → clique simples
                        _fire(_click_single, gaze_x, gaze_y,
                              f'CLIQUE SIMPLES #{self._n} ({duracao_ms:.0f} ms)')
                        print(f'[BlinkClicker] EAR threshold={BLINK_EAR_THRESHOLD} '
                            f'| duração={duracao_ms:.0f}ms')
                        self._t_last_click = now
                        self._last_x = gaze_x
                        self._last_y = gaze_y
                        self._cooldown = now
                        return 'single'

                elif duracao_ms <= BLINK_RIGHT_MAX_MS:
                    # Piscada longa → clique direito
                    _fire(_click_right, gaze_x, gaze_y,
                          f'CLIQUE DIREITO #{self._n} ({duracao_ms:.0f} ms)')
                    print(f'[BlinkClicker] EAR threshold={BLINK_EAR_THRESHOLD} '
                        f'| duração={duracao_ms:.0f}ms')
                    self._t_last_click = 0.0
                    self._cooldown = now
                    return 'right'

                else:
                    # Duração excessiva (olho fechado demais) → ignora
                    pass

        return None

    @property
    def eye_closed(self) -> bool:
        return self._eye_closed

    def reset(self):
        self._eye_closed   = False
        self._t_fechou     = 0.0
        self._t_last_click = 0.0
        self._cooldown     = 0.0