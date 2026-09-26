"""
EyeNav – Sistema de Navegação Web por Rastreamento Ocular

Trabalho de Curso II – Ciência da Computação  |  UNIP EAD  |  2026
Autor  : Erick Martins Paulino Silva
Tema   : Navegação pela internet por rastreamento ocular como solução
         de acessibilidade para pessoas com imobilidade física
Linha  : Tecnologia e Bem-Estar Social

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline completo (por frame):
  ① Captura de frame (webcam)
  ② Detecção de malha facial — MediaPipe Face Mesh
  ③ Extração da posição normalizada da íris (GazeEstimator)
  ④ Mapeamento gaze → tela — modelo polinomial (GazeModel)
  ⑤ Suavização EMA do ponto de olhar (EMASmoother)
  ⑥ Movimento do cursor do sistema (CursorController)
  ⑦ Detecção e disparo de dwell click (DwellClicker)
  ⑧ Coleta de métricas (MetricsCollector)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Controles (janela de debug):
  Q / ESC  → Encerrar
  R        → Recalibrar
  D        → Ativar / desativar dwell click
  M        → Salvar métricas da sessão em CSV
  S        → Salvar screenshot do debug
"""

import sys
import time
import cv2
import numpy as np

# Resolução da tela
try:
    from screeninfo import get_monitors
    _m = get_monitors()[0]
    SCREEN_W, SCREEN_H = _m.width, _m.height
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080

# Módulos do projeto
from config           import (WEBCAM_INDEX, WEBCAM_WIDTH, WEBCAM_HEIGHT,
                               TARGET_FPS, CALIB_FILE,
                               SHOW_DEBUG_WINDOW, DEBUG_WINDOW_SCALE,
                               DWELL_RADIUS_PX)
from gaze_estimator   import GazeEstimator
from gaze_model       import GazeModel
from calibration      import CalibrationScreen
from smoother         import EMASmoother
from cursor_controller import CursorController
from dwell_clicker    import DwellClicker
from metrics          import MetricsCollector


# Constantes visuais
_GREEN  = (0, 220, 80)
_CYAN   = (0, 220, 220)
_RED    = (0, 60, 220)
_WHITE  = (220, 220, 220)
_GRAY   = (120, 120, 120)
_YELLOW = (0, 200, 255)


# Helpers de desenho

def _draw_dwell_arc(frame: np.ndarray, cx: int, cy: int, progress: float):
    """Desenha indicador circular de progresso do dwell click."""
    r = DWELL_RADIUS_PX
    cv2.circle(frame, (cx, cy), r, _GRAY, 1)
    if progress > 0:
        angle = int(360 * progress)
        for a in range(0, min(angle, 360), 4):
            rad = np.radians(a - 90)
            px  = int(cx + (r - 4) * np.cos(rad))
            py  = int(cy + (r - 4) * np.sin(rad))
            g   = int(80  + 140 * progress)
            b   = int(200 * progress)
            cv2.circle(frame, (px, py), 2, (b, g, 0), -1)
    dot_color = _GREEN if progress < 1.0 else (0, 255, 0)
    cv2.circle(frame, (cx, cy), 7, dot_color, -1)


def _draw_info_panel(frame: np.ndarray, lines: list[tuple[str, tuple]]):
    """Painel de informações no canto superior esquerdo."""
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (8, 28 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


# Aplicação principal

def main():
    _banner()

    # Módulos
    estimator = GazeEstimator()
    model     = GazeModel(degree=2, alpha=1.0)
    smoother  = EMASmoother()
    cursor    = CursorController(SCREEN_W, SCREEN_H)
    dwell     = DwellClicker()
    metrics   = MetricsCollector()

    # Câmera
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)

    if not cap.isOpened():
        print(f'[ERRO] Câmera {WEBCAM_INDEX} não encontrada.')
        print('       Verifique WEBCAM_INDEX em config.py')
        sys.exit(1)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'  Câmera aberta: {cam_w}×{cam_h} @ índice {WEBCAM_INDEX}')
    print(f'  Tela:          {SCREEN_W}×{SCREEN_H}')

    # Calibração
    _run_calibration_if_needed(model, estimator, cap)

    # Janela de debug
    debug_win = None
    if SHOW_DEBUG_WINDOW:
        debug_win = 'EyeNav – Debug'
        cv2.namedWindow(debug_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(debug_win,
                         int(cam_w * DEBUG_WINDOW_SCALE),
                         int(cam_h * DEBUG_WINDOW_SCALE))
        # Impede que a janela de debug roube o foco do browser
        cv2.setWindowProperty(debug_win, cv2.WND_PROP_TOPMOST, 1)
        # Move para canto inferior direito para não obstruir a navegação
        try:
            from screeninfo import get_monitors
            _m = get_monitors()[0]
            _dw = int(cam_w * DEBUG_WINDOW_SCALE)
            _dh = int(cam_h * DEBUG_WINDOW_SCALE)
            cv2.moveWindow(debug_win, _m.width - _dw - 10, _m.height - _dh - 60)
        except Exception:
            cv2.moveWindow(debug_win, 10, 10)

    _print_controls()

    # Estado
    dwell_on    = True
    gaze_sx     = SCREEN_W // 2
    gaze_sy     = SCREEN_H // 2
    fps_val     = 0.0
    fps_counter = 0
    fps_t       = time.time()
    screenshot_n = 0

    # Loop principal
    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        fps_counter += 1

        # FPS a cada 30 frames
        if fps_counter % 30 == 0:
            fps_val     = 30 / max(time.time() - fps_t, 1e-9)
            fps_t       = time.time()
            fps_counter = 0

        # Estimativa do olhar
        feat = estimator.process(frame)
        face_ok = feat is not None

        if face_ok and model.is_trained:
            raw_x, raw_y = model.predict(feat['gaze_vector'])
            sx, sy       = smoother.update(raw_x, raw_y)
            gaze_sx      = int(max(0, min(sx, SCREEN_W - 1)))
            gaze_sy      = int(max(0, min(sy, SCREEN_H - 1)))

            cursor.move(gaze_sx, gaze_sy)

            if dwell_on:
                clicked = dwell.update(gaze_sx, gaze_sy)
                if clicked:
                    metrics.record_click(gaze_sx, gaze_sy,
                                         dwell_ms=1500)
                    print(f'  [CLICK] ({gaze_sx}, {gaze_sy})')

        # Latência por frame
        dt_ms = (time.time() - t0) * 1000
        metrics.record_frame(dt_ms)

        # Janela de debug
        if debug_win:
            dbg = frame.copy()

            # Overlays de estimativa
            if face_ok:
                dbg = estimator.draw_debug(dbg, feat)

            # Indicador de dwell
            if dwell_on and face_ok:
                prog = dwell.progress
                # Projeta ponto de tela de volta ao frame (escala)
                fx = int(gaze_sx * cam_w / SCREEN_W)
                fy = int(gaze_sy * cam_h / SCREEN_H)
                _draw_dwell_arc(dbg, fx, fy, prog)

            # Painel de info
            status_face  = 'Detectada' if face_ok else 'Ausente'
            status_calib = 'Sim'       if model.is_trained else 'Não'
            status_dwell = 'ON'        if dwell_on else 'OFF'
            color_face   = _GREEN if face_ok else _RED

            info = [
                (f'FPS: {fps_val:.1f}  |  Latência: {dt_ms:.1f} ms', _CYAN),
                (f'Face: {status_face}', color_face),
                (f'Calibrado: {status_calib}', _WHITE),
                (f'Dwell: {status_dwell}', _GREEN if dwell_on else _GRAY),
                (f'Gaze (tela): ({gaze_sx}, {gaze_sy})', _YELLOW),
            ]
            if dwell_on and face_ok:
                info.append((f'Progresso dwell: {dwell.progress*100:.0f}%', _YELLOW))

            _draw_info_panel(dbg, info)

            # Rodapé
            cv2.putText(dbg, 'Q=Sair  R=Recalibrar  D=Dwell  M=Metricas  S=Screenshot',
                        (6, cam_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GRAY, 1)

            cv2.imshow(debug_win, dbg)

        # Teclas
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key == ord('r'):
            print('\n  Iniciando recalibração...')
            _do_recalibration(model, estimator, cap, smoother, dwell)

        elif key == ord('d'):
            dwell_on = not dwell_on
            dwell.reset()
            print(f'  Dwell click: {"ATIVADO" if dwell_on else "DESATIVADO"}')

        elif key == ord('m'):
            metrics.print_summary()
            metrics.save_csv(f'eyenav_metrics.csv')

        elif key == ord('s'):
            if debug_win:
                fname = f'eyenav_screenshot_{screenshot_n:03d}.png'
                cv2.imwrite(fname, dbg)
                print(f'  Screenshot salvo: {fname}')
                screenshot_n += 1

    # Encerramento
    print('\n  Encerrando EyeNav...')
    metrics.print_summary()
    metrics.save_csv()
    cap.release()
    cv2.destroyAllWindows()
    estimator.close()
    print('  Encerrado. Até a próxima!')


# Funções auxiliares

def _run_calibration_if_needed(model, estimator, cap):
    """Carrega calibração existente ou solicita nova ao usuário."""
    if model.load(CALIB_FILE):
        resp = input('\n  Calibração encontrada. Deseja recalibrar? (s/N): ')
        if resp.strip().lower() != 's':
            return

    print('\n┌────────────────────────────────────────────┐')
    print('  │  CALIBRAÇÃO                                │')
    print('  │  Fixe o olhar em cada círculo até          │')
    print('  │  o arco completar (≈ 1,5 s por ponto).     │')
    print('  │  ESC = cancelar                            │')
    print('  └────────────────────────────────────────────┘')
    input('  Pressione ENTER para iniciar...')

    from screeninfo import get_monitors
    try:
        m = get_monitors()[0]
        sw, sh = m.width, m.height
    except Exception:
        sw, sh = 1920, 1080

    screen   = CalibrationScreen(sw, sh)
    features, targets = screen.run(estimator, cap)

    if not features:
        print('  [AVISO] Calibração cancelada ou incompleta.')
        return

    model.train(features, targets)
    model.save(CALIB_FILE)
    print(f'  Calibração concluída com {len(features)} pontos.\n')


def _do_recalibration(model, estimator, cap, smoother, dwell):
    """Recalibra e atualiza modelo/suavizador/dwell."""
    from screeninfo import get_monitors
    try:
        m = get_monitors()[0]
        sw, sh = m.width, m.height
    except Exception:
        sw, sh = 1920, 1080

    screen   = CalibrationScreen(sw, sh)
    features, targets = screen.run(estimator, cap)

    if features:
        model.train(features, targets)
        model.save(CALIB_FILE)
        smoother.reset()
        dwell.reset()
        print(f'  Recalibração concluída ({len(features)} pontos).\n')
    else:
        print('  Recalibração cancelada.\n')


def _banner():
    print()
    print('  ╔══════════════════════════════════════════════════════╗')
    print('  ║          EyeNav – Navegação por Rastreamento         ║')
    print('  ║          Ocular para Acessibilidade Digital          ║')
    print('  ║          UNIP EAD  |  TC II  |  2026                 ║')
    print('  ╚══════════════════════════════════════════════════════╝')
    print()


def _print_controls():
    print()
    print('  ┌───────────────────────────────────────────────────┐')
    print('  │  SISTEMA ATIVO – Controles na janela de debug:    │')
    print('  │   Q / ESC → Encerrar                              │')
    print('  │   R       → Recalibrar                            │')
    print('  │   D       → Ativar / desativar dwell click        │')
    print('  │   M       → Exibir e salvar métricas (CSV)        │')
    print('  │   S       → Salvar screenshot do debug            │')
    print('  └───────────────────────────────────────────────────┘')
    print()


if __name__ == '__main__':
    main()