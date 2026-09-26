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
  7. Novo movimento além do raio reinicia o ciclo.

Referências:
  Sibert & Jacob (2000) — primeira avaliação sistemática de dwell time
  como mecanismo de seleção em interfaces gráficas (ACM CHI).
  Majaranta & Bulling (2014) — revisão de técnicas de seleção ocular.
"""

import os
import time
import subprocess
import threading
from config import DWELL_TIME_MS, DWELL_RADIUS_PX

# Detecção de ambiente e backend

_SESSION  = os.environ.get('XDG_SESSION_TYPE', 'x11').lower()
_WAYLAND  = bool(os.environ.get('WAYLAND_DISPLAY'))
_DISPLAY  = os.environ.get('DISPLAY', ':0')

def _cmd_ok(cmd: str) -> bool:
    """Verifica se um comando está disponível no sistema."""
    try:
        r = subprocess.run(['which', cmd],
                           capture_output=True, timeout=2)
        return r.returncode == 0
    except Exception:
        return False

_HAS_XDOTOOL = _cmd_ok('xdotool')
_HAS_YDOTOOL = _cmd_ok('ydotool')

try:
    from pynput.mouse import Controller as _MC, Button as _Btn
    _pynput_mouse = _MC()
    _HAS_PYNPUT = True
except Exception:
    _pynput_mouse = None
    _HAS_PYNPUT = False

# Escolhe backend
if _WAYLAND and _HAS_YDOTOOL:
    _BACKEND = 'ydotool'
elif _HAS_XDOTOOL:
    _BACKEND = 'xdotool'
elif _HAS_PYNPUT:
    _BACKEND = 'pynput'
else:
    _BACKEND = 'none'

print(f'[DwellClicker] Ambiente: {_SESSION}'
      f'{"(Wayland)" if _WAYLAND else "(X11)"}')
print(f'[DwellClicker] Backend de clique: {_BACKEND}')
if _BACKEND == 'none':
    print('[DwellClicker] AVISO: nenhum backend de clique disponível!')
    print('  Instale xdotool:  sudo apt install xdotool')
    print('  Ou ydotool (Wayland): sudo apt install ydotool')


# Função de clique

def _do_click(x: int, y: int):
    """
    Dispara clique esquerdo na posição (x, y).
    Executado em thread separada para não bloquear o loop de frames.
    """
    def _click():
        try:
            if _BACKEND == 'xdotool':
                # mousemove + click em um comando só
                subprocess.run(
                    ['xdotool', 'mousemove', '--sync',
                     str(x), str(y), 'click', '--clearmodifiers', '1'],
                    timeout=1, capture_output=True,
                    env={**os.environ, 'DISPLAY': _DISPLAY}
                )

            elif _BACKEND == 'ydotool':
                # ydotool para Wayland
                subprocess.run(
                    ['ydotool', 'mousemove', '--absolute',
                     f'-x', str(x), f'-y', str(y)],
                    timeout=1, capture_output=True
                )
                time.sleep(0.05)
                subprocess.run(
                    ['ydotool', 'click', '0x40001'],   # botão esquerdo
                    timeout=1, capture_output=True
                )

            elif _BACKEND == 'pynput' and _pynput_mouse:
                _pynput_mouse.position = (x, y)
                time.sleep(0.02)
                _pynput_mouse.press(_Btn.left)
                time.sleep(0.02)
                _pynput_mouse.release(_Btn.left)

        except Exception as e:
            print(f'[DwellClicker] Erro no clique: {e}')

    threading.Thread(target=_click, daemon=True).start()


# Classe principal

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

        if dist > DWELL_RADIUS_PX:
            self._reset_anchor(gx, gy)
            return False

        if self._fired:
            return False

        elapsed_ms = (time.time() - self._t_start) * 1000
        if elapsed_ms >= DWELL_TIME_MS:
            ax, ay = int(self._ax), int(self._ay)
            self._click_count += 1
            print(f'[DwellClicker] CLIQUE #{self._click_count} '
                  f'em ({ax}, {ay}) — backend: {_BACKEND}')
            _do_click(ax, ay)
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