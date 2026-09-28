"""
Controle do cursor do sistema operacional.

Restringe o movimento aos limites da tela e delega a execução
ao backend unificado (mouse_backend).
"""

import mouse_backend
from config import CURSOR_MOVE


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
        mouse_backend.move(cx, cy)

    def scroll(self, clicks: int):
        """clicks > 0 = para cima; clicks < 0 = para baixo."""
        mouse_backend.scroll(clicks)