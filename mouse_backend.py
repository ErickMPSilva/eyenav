"""
Backend unificado de controle do mouse.

Centraliza a detecção do mecanismo de automação disponível no sistema
e expõe uma API única para mover o cursor, clicar e rolar a página.
Os módulos de interação (CursorController, DwellClicker, BlinkClicker,
EdgeScroller) usam apenas este módulo, sem duplicar código de backend.

Ordem de escolha do backend:
  1. ydotool — sessões Wayland (requer o daemon ydotoold em execução)
  2. xdotool — sessões X11
  3. pynput  — fallback multiplataforma (X11, Windows, macOS)
  4. none    — nenhum disponível; as ações são ignoradas

API pública:
  BACKEND                 nome do backend em uso
  available()             True se há backend funcional
  move(x, y)              move o cursor (não bloqueante)
  click(x, y)             clique esquerdo
  double_click(x, y)      duplo clique esquerdo
  right_click(x, y)       clique direito
  scroll(clicks)          rolagem (+ = cima, − = baixo)

Cliques e scroll rodam em thread separada para não bloquear
o loop de processamento de frames.
"""

import os
import shutil
import subprocess
import threading
import time

# Detecção de ambiente

_SESSION = os.environ.get('XDG_SESSION_TYPE', '').lower() or 'desconhecida'
_WAYLAND = bool(os.environ.get('WAYLAND_DISPLAY'))
_DISPLAY = os.environ.get('DISPLAY', ':0')
_ENV     = {**os.environ, 'DISPLAY': _DISPLAY}

_HAS_XDOTOOL = shutil.which('xdotool') is not None
_HAS_YDOTOOL = shutil.which('ydotool') is not None

try:
    from pynput.mouse import Controller as _Controller, Button as _Button
    _pm = _Controller()
    _HAS_PYNPUT = True
except Exception:
    _pm = None
    _Button = None
    _HAS_PYNPUT = False

if _WAYLAND and _HAS_YDOTOOL:
    BACKEND = 'ydotool'
elif _HAS_XDOTOOL:
    BACKEND = 'xdotool'
elif _HAS_PYNPUT:
    BACKEND = 'pynput'
else:
    BACKEND = 'none'

print(f'[MouseBackend] Sessão: {_SESSION} '
      f'({"Wayland" if _WAYLAND else "X11/outra"}) | Backend: {BACKEND}')
if BACKEND == 'none':
    print('[MouseBackend] AVISO: nenhum backend de mouse disponível!')
    print('  X11:     sudo apt install xdotool')
    print('  Wayland: sudo apt install ydotool  (e inicie o ydotoold)')
    print('  Qualquer SO: pip install pynput')


def available() -> bool:
    """Indica se existe um backend capaz de controlar o mouse."""
    return BACKEND != 'none'


# Helpers internos

def _run(cmd: list[str]):
    """Executa um comando e aguarda o término (usado dentro de threads)."""
    try:
        subprocess.run(cmd, timeout=1, capture_output=True, env=_ENV)
    except Exception as e:
        print(f'[MouseBackend] Erro ao executar {cmd[0]}: {e}')


def _spawn(cmd: list[str]):
    """Dispara um comando sem aguardar (usado no movimento contínuo)."""
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, env=_ENV)
    except Exception:
        pass


def _in_thread(fn, *args):
    threading.Thread(target=fn, args=args, daemon=True).start()


# Movimento

def move(x: int, y: int):
    """Move o cursor para (x, y). Chamado a cada frame — não bloqueia."""
    x, y = int(x), int(y)
    try:
        if BACKEND == 'xdotool':
            _spawn(['xdotool', 'mousemove', str(x), str(y)])
        elif BACKEND == 'ydotool':
            _spawn(['ydotool', 'mousemove', '--absolute',
                    '-x', str(x), '-y', str(y)])
        elif BACKEND == 'pynput':
            _pm.position = (x, y)
    except Exception:
        pass


# Cliques

def _click_sync(x: int, y: int, button: str, count: int):
    """Move até (x, y) e clica `count` vezes com o botão indicado."""
    x, y = int(x), int(y)
    try:
        if BACKEND == 'xdotool':
            btn = '1' if button == 'left' else '3'
            # Sem '--sync': ele espera o cursor se MOVER e trava quando o
            # cursor já está no destino (caso comum: o cursor segue o olhar)
            _run(['xdotool', 'mousemove', str(x), str(y)])
            time.sleep(0.03)
            _run(['xdotool', 'click', '--clearmodifiers',
                  '--repeat', str(count), '--delay', '80', btn])

        elif BACKEND == 'ydotool':
            # Sintaxe do ydotool 1.x: 0xC0 = clique esquerdo (down+up),
            # 0xC1 = clique direito (down+up)
            code = '0xC0' if button == 'left' else '0xC1'
            _run(['ydotool', 'mousemove', '--absolute',
                  '-x', str(x), '-y', str(y)])
            time.sleep(0.05)
            for i in range(count):
                if i:
                    time.sleep(0.08)
                _run(['ydotool', 'click', code])

        elif BACKEND == 'pynput':
            _pm.position = (x, y)
            time.sleep(0.02)
            btn = _Button.left if button == 'left' else _Button.right
            _pm.click(btn, count)

    except Exception as e:
        print(f'[MouseBackend] Erro no clique: {e}')


def click(x: int, y: int):
    """Clique esquerdo simples em (x, y)."""
    if available():
        _in_thread(_click_sync, x, y, 'left', 1)


def double_click(x: int, y: int):
    """Duplo clique esquerdo em (x, y)."""
    if available():
        _in_thread(_click_sync, x, y, 'left', 2)


def right_click(x: int, y: int):
    """Clique direito em (x, y)."""
    if available():
        _in_thread(_click_sync, x, y, 'right', 1)


# Scroll

def _scroll_sync(clicks: int):
    try:
        if BACKEND == 'xdotool':
            # Botão 4 = roda para cima, 5 = roda para baixo
            btn = '4' if clicks > 0 else '5'
            _run(['xdotool', 'click', '--repeat', str(abs(clicks)), btn])

        elif BACKEND == 'ydotool':
            # ydotool 1.x: rolagem via --wheel (valide o sinal no seu sistema)
            _run(['ydotool', 'mousemove', '--wheel',
                  '-x', '0', '-y', str(clicks)])

        elif BACKEND == 'pynput':
            _pm.scroll(0, clicks)

    except Exception as e:
        print(f'[MouseBackend] Erro no scroll: {e}')


def scroll(clicks: int):
    """Rola a página. clicks > 0 = para cima; clicks < 0 = para baixo."""
    if available() and clicks:
        _in_thread(_scroll_sync, int(clicks))