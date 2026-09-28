"""
Marcador visual do olhar: bola vermelha sobre a tela.

Cria uma janela X11 minúscula, redonda, sempre por cima de tudo, que
acompanha o ponto de olhar. Duas propriedades são essenciais:

  - Formato circular: via extensão SHAPE (região "Bounding" = círculo).
  - Transparente a cliques: região "Input" vazia. Sem isso, o dwell
    click cairia na própria bola, e não na página embaixo dela.

A janela é "override-redirect": o gerenciador de janelas não a decora,
não a lista na barra de tarefas e não lhe dá foco.

Implementado com ctypes sobre libX11/libXext (presentes em qualquer
desktop Linux X11), sem dependências Python extras. Em Wayland,
Windows ou macOS o marcador é desativado com um aviso; o resto do
sistema funciona normalmente.
"""

import ctypes
import ctypes.util
import os

from config import GAZE_DOT_ENABLED, GAZE_DOT_RADIUS, GAZE_DOT_COLOR

# Tipos e constantes do Xlib
_Window  = ctypes.c_ulong
_Pixmap  = ctypes.c_ulong
_GC      = ctypes.c_void_p
_Display = ctypes.c_void_p

_CW_BACK_PIXEL        = 1 << 1
_CW_BORDER_PIXEL      = 1 << 3
_CW_OVERRIDE_REDIRECT = 1 << 9
_INPUT_OUTPUT         = 1
_SHAPE_SET            = 0
_SHAPE_BOUNDING       = 0
_SHAPE_INPUT          = 2
_UNSORTED             = 0


class _XSetWindowAttributes(ctypes.Structure):
    _fields_ = [
        ('background_pixmap', ctypes.c_ulong),
        ('background_pixel',  ctypes.c_ulong),
        ('border_pixmap',     ctypes.c_ulong),
        ('border_pixel',      ctypes.c_ulong),
        ('bit_gravity',       ctypes.c_int),
        ('win_gravity',       ctypes.c_int),
        ('backing_store',     ctypes.c_int),
        ('backing_planes',    ctypes.c_ulong),
        ('backing_pixel',     ctypes.c_ulong),
        ('save_under',        ctypes.c_int),
        ('event_mask',        ctypes.c_long),
        ('do_not_propagate_mask', ctypes.c_long),
        ('override_redirect', ctypes.c_int),
        ('colormap',          ctypes.c_ulong),
        ('cursor',            ctypes.c_ulong),
    ]


class _XColor(ctypes.Structure):
    _fields_ = [('pixel', ctypes.c_ulong),
                ('red',   ctypes.c_ushort),
                ('green', ctypes.c_ushort),
                ('blue',  ctypes.c_ushort),
                ('flags', ctypes.c_char),
                ('pad',   ctypes.c_char)]


def _load_libs():
    x11  = ctypes.util.find_library('X11')  or 'libX11.so.6'
    xext = ctypes.util.find_library('Xext') or 'libXext.so.6'
    X  = ctypes.CDLL(x11)
    Xe = ctypes.CDLL(xext)

    X.XOpenDisplay.restype      = _Display
    X.XOpenDisplay.argtypes     = [ctypes.c_char_p]
    X.XDefaultScreen.argtypes   = [_Display]
    X.XRootWindow.restype       = _Window
    X.XRootWindow.argtypes      = [_Display, ctypes.c_int]
    X.XDefaultColormap.restype  = ctypes.c_ulong
    X.XDefaultColormap.argtypes = [_Display, ctypes.c_int]
    X.XAllocColor.argtypes      = [_Display, ctypes.c_ulong,
                                   ctypes.POINTER(_XColor)]
    X.XCreateWindow.restype     = _Window
    X.XCreateWindow.argtypes    = [_Display, _Window, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                   ctypes.c_int, ctypes.c_uint, ctypes.c_void_p,
                                   ctypes.c_ulong,
                                   ctypes.POINTER(_XSetWindowAttributes)]
    X.XCreatePixmap.restype     = _Pixmap
    X.XCreatePixmap.argtypes    = [_Display, _Window, ctypes.c_uint,
                                   ctypes.c_uint, ctypes.c_uint]
    X.XCreateGC.restype         = _GC
    X.XCreateGC.argtypes        = [_Display, ctypes.c_ulong, ctypes.c_ulong,
                                   ctypes.c_void_p]
    X.XSetForeground.argtypes   = [_Display, _GC, ctypes.c_ulong]
    X.XFillRectangle.argtypes   = [_Display, ctypes.c_ulong, _GC, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
    X.XFillArc.argtypes         = [_Display, ctypes.c_ulong, _GC, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                                   ctypes.c_int, ctypes.c_int]
    X.XFreeGC.argtypes          = [_Display, _GC]
    X.XFreePixmap.argtypes      = [_Display, _Pixmap]
    for fn in ('XMapRaised', 'XUnmapWindow', 'XRaiseWindow', 'XDestroyWindow'):
        getattr(X, fn).argtypes = [_Display, _Window]
    X.XMoveWindow.argtypes      = [_Display, _Window, ctypes.c_int, ctypes.c_int]
    X.XFlush.argtypes           = [_Display]
    X.XCloseDisplay.argtypes    = [_Display]

    Xe.XShapeQueryExtension.argtypes = [_Display, ctypes.POINTER(ctypes.c_int),
                                        ctypes.POINTER(ctypes.c_int)]
    Xe.XShapeCombineMask.argtypes = [_Display, _Window, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, _Pixmap,
                                     ctypes.c_int]
    Xe.XShapeCombineRectangles.argtypes = [_Display, _Window, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int]
    return X, Xe


class GazeOverlay:
    """Bola vermelha que segue o olhar, sem interceptar cliques."""

    def __init__(self, radius: int = GAZE_DOT_RADIUS,
                 color: tuple = GAZE_DOT_COLOR):
        self.r = radius
        self._ok = False
        self._visible = False

        if not GAZE_DOT_ENABLED:
            return
        if os.environ.get('WAYLAND_DISPLAY') or not os.environ.get('DISPLAY'):
            print('[GazeOverlay] Marcador disponível apenas em X11 — desativado.')
            return
        try:
            self._init_x11(color)
            self._ok = True
            print(f'[GazeOverlay] Marcador do olhar ativo (raio {radius} px).')
        except Exception as e:
            print(f'[GazeOverlay] Não foi possível criar o marcador: {e}')

    def _init_x11(self, color):
        X, Xe = _load_libs()
        self.X = X
        dpy = X.XOpenDisplay(None)
        if not dpy:
            raise RuntimeError('XOpenDisplay falhou')
        self.dpy = dpy

        ev, er = ctypes.c_int(), ctypes.c_int()
        if not Xe.XShapeQueryExtension(dpy, ctypes.byref(ev), ctypes.byref(er)):
            raise RuntimeError('extensão SHAPE indisponível')

        scr  = X.XDefaultScreen(dpy)
        root = X.XRootWindow(dpy, scr)

        # Cor (RGB 0–255 → 0–65535)
        xc = _XColor()
        xc.red, xc.green, xc.blue = (c * 257 for c in color)
        X.XAllocColor(dpy, X.XDefaultColormap(dpy, scr), ctypes.byref(xc))

        size  = 2 * self.r
        attrs = _XSetWindowAttributes()
        attrs.background_pixel  = xc.pixel
        attrs.border_pixel      = 0
        attrs.override_redirect = 1
        mask = _CW_BACK_PIXEL | _CW_BORDER_PIXEL | _CW_OVERRIDE_REDIRECT

        self.win = X.XCreateWindow(dpy, root, 0, 0, size, size, 0, 0,
                                   _INPUT_OUTPUT, None, mask,
                                   ctypes.byref(attrs))

        # Formato circular: bitmap de 1 bit com um disco
        pm = X.XCreatePixmap(dpy, self.win, size, size, 1)
        gc = X.XCreateGC(dpy, pm, 0, None)
        X.XSetForeground(dpy, gc, 0)
        X.XFillRectangle(dpy, pm, gc, 0, 0, size, size)
        X.XSetForeground(dpy, gc, 1)
        X.XFillArc(dpy, pm, gc, 0, 0, size, size, 0, 360 * 64)
        Xe.XShapeCombineMask(dpy, self.win, _SHAPE_BOUNDING, 0, 0, pm, _SHAPE_SET)
        X.XFreeGC(dpy, gc)
        X.XFreePixmap(dpy, pm)

        # Transparente a cliques: região de entrada vazia
        Xe.XShapeCombineRectangles(dpy, self.win, _SHAPE_INPUT, 0, 0,
                                   None, 0, _SHAPE_SET, _UNSORTED)
        X.XFlush(dpy)

    # API

    def move(self, x: int, y: int):
        if not self._ok:
            return
        if not self._visible:
            self.show()
        self.X.XMoveWindow(self.dpy, self.win, int(x) - self.r, int(y) - self.r)
        self.X.XRaiseWindow(self.dpy, self.win)
        self.X.XFlush(self.dpy)

    def show(self):
        if self._ok and not self._visible:
            self.X.XMapRaised(self.dpy, self.win)
            self.X.XFlush(self.dpy)
            self._visible = True

    def hide(self):
        if self._ok and self._visible:
            self.X.XUnmapWindow(self.dpy, self.win)
            self.X.XFlush(self.dpy)
            self._visible = False

    def close(self):
        if self._ok:
            self.X.XDestroyWindow(self.dpy, self.win)
            self.X.XCloseDisplay(self.dpy)
            self._ok = False