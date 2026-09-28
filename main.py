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
  ③ Extração da posição normalizada da íris + EAR (GazeEstimator)
  ④ Mapeamento gaze → tela — modelo polinomial (GazeModel)
  ⑤ Suavização do olhar: mediana + filtro One Euro (GazeSmoother)
  ⑥ Ganho + atração de borda (alcance dos cantos da tela)
  ⑦ Movimento do cursor do sistema (CursorController)
  ⑧ Detecção e disparo de dwell click (DwellClicker)
  ⑨ Detecção de piscada intencional (BlinkClicker)
  ⑩ Scroll por borda da tela (EdgeScroller)
  ⑪ Coleta de métricas (MetricsCollector)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Controles (janela de debug):
  Q / ESC  → Encerrar
  R        → Recalibrar
  F        → Refazer ajuste fino
  D        → Ativar / desativar dwell click
  B        → Ativar / desativar blink click
  M        → Salvar métricas da sessão em CSV
  S        → Salvar screenshot do debug
"""

import sys
import time
import cv2
import numpy as np
import mouse_backend

# Resolução da tela
try:
    from screeninfo import get_monitors
    _m = get_monitors()[0]
    SCREEN_W, SCREEN_H = _m.width, _m.height
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080

# Módulos do projeto
from config            import (WEBCAM_INDEX, WEBCAM_WIDTH, WEBCAM_HEIGHT,
                                TARGET_FPS, CALIB_FILE,
                                SHOW_DEBUG_WINDOW, DEBUG_WINDOW_SCALE,
                                DWELL_TIME_MS, DWELL_RADIUS_PX,
                                BLINK_CLICK_ENABLED, FINE_TUNE_ENABLED,
                                CURSOR_GAIN_X, CURSOR_GAIN_Y, EDGE_SNAP_PX)
from gaze_estimator    import GazeEstimator
from gaze_model        import GazeModel
from calibration       import CalibrationScreen
from fine_tune         import FineTuneScreen
from gaze_overlay      import GazeOverlay
from smoother          import GazeSmoother
from cursor_controller import CursorController
from dwell_clicker     import DwellClicker
from blink_clicker     import BlinkClicker
from eye_state         import EyeState
from edge_scroller     import EdgeScroller
from metrics           import MetricsCollector


# Constantes visuais
_GREEN  = (0, 220, 80)
_CYAN   = (0, 220, 220)
_RED    = (0, 60, 220)
_WHITE  = (220, 220, 220)
_GRAY   = (120, 120, 120)
_YELLOW = (0, 200, 255)
_ORANGE = (0, 165, 255)


# Mapeamento final do cursor

def _to_screen(x: float, y: float) -> tuple[int, int]:
    """
    Aplica ganho a partir do centro e atração de borda.

    Ganho (CURSOR_GAIN_X/Y): compensa a leve compressão do modelo nas
    extremidades, onde a estimativa da íris é menos precisa.
    Atração de borda (EDGE_SNAP_PX): perto da borda, o cursor vai
    exatamente até ela — necessário para alcançar cantos, barras
    de rolagem e as zonas do edge scroll.
    """
    cx, cy = SCREEN_W / 2, SCREEN_H / 2
    x = cx + (x - cx) * CURSOR_GAIN_X
    y = cy + (y - cy) * CURSOR_GAIN_Y

    if x < EDGE_SNAP_PX:
        x = 0
    elif x > SCREEN_W - 1 - EDGE_SNAP_PX:
        x = SCREEN_W - 1
    if y < EDGE_SNAP_PX:
        y = 0
    elif y > SCREEN_H - 1 - EDGE_SNAP_PX:
        y = SCREEN_H - 1

    return (int(max(0, min(x, SCREEN_W - 1))),
            int(max(0, min(y, SCREEN_H - 1))))


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


def _draw_scroll_zone(frame: np.ndarray, cam_h: int, cam_w: int,
                      zone: str | None):
    """Destaca visualmente a borda de scroll ativa."""
    if zone == 'top':
        cv2.rectangle(frame, (0, 0), (cam_w, 18), _ORANGE, -1)
        cv2.putText(frame, 'SCROLL UP', (cam_w // 2 - 45, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    elif zone == 'bottom':
        cv2.rectangle(frame, (0, cam_h - 18), (cam_w, cam_h), _ORANGE, -1)
        cv2.putText(frame, 'SCROLL DOWN', (cam_w // 2 - 55, cam_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def _draw_info_panel(frame: np.ndarray, lines: list[tuple[str, tuple]]):
    """Painel de informações no canto superior esquerdo."""
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (8, 28 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


# Aplicação principal

def main():
    _banner()

    if not mouse_backend.available():
        print('[ERRO] Nenhum backend de mouse disponível.')
        print('       sudo apt install xdotool   ou   pip install pynput')
        sys.exit(1)

    # Módulos
    estimator = GazeEstimator()
    model     = GazeModel(degree=2, alpha=1.0)
    smoother  = GazeSmoother()
    cursor    = CursorController(SCREEN_W, SCREEN_H)
    dwell     = DwellClicker()
    blink     = BlinkClicker() if BLINK_CLICK_ENABLED else None
    scroller  = EdgeScroller(screen_h=SCREEN_H)
    eye_state = EyeState()
    metrics   = MetricsCollector()

    # Câmera
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    # MJPG: muitas webcams só entregam 720p/30fps nesse formato;
    # em YUYV caem para 640×480, deixando o olho com ~30 px na imagem
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
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
    if cam_w < WEBCAM_WIDTH:
        print(f'  [AVISO] Resolução abaixo da pedida ({WEBCAM_WIDTH}×{WEBCAM_HEIGHT}). '
              f'A precisão do olhar será menor.')
    print(f'  Tela:          {SCREEN_W}×{SCREEN_H}')

    # Calibração
    _run_calibration_if_needed(model, estimator, cap)

    # Ajuste fino: viés + filtro calibrados para a sessão atual
    bias = _fine_tune(model, estimator, cap, smoother)

    # Bola vermelha no ponto de olhar
    overlay = GazeOverlay()

    # Janela de debug
    debug_win = None
    if SHOW_DEBUG_WINDOW:
        debug_win = 'EyeNav – Debug'
        cv2.namedWindow(debug_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(debug_win,
                         int(cam_w * DEBUG_WINDOW_SCALE),
                         int(cam_h * DEBUG_WINDOW_SCALE))
        cv2.setWindowProperty(debug_win, cv2.WND_PROP_TOPMOST, 1)
        _dw = int(cam_w * DEBUG_WINDOW_SCALE)
        _dh = int(cam_h * DEBUG_WINDOW_SCALE)
        cv2.moveWindow(debug_win, SCREEN_W - _dw - 10, SCREEN_H - _dh - 60)

    _print_controls()

    # Estado
    dwell_on     = True
    blink_on     = bool(blink)
    gaze_sx      = SCREEN_W // 2
    gaze_sy      = SCREEN_H // 2
    fps_val      = 0.0
    fps_counter  = 0
    fps_t        = time.time()
    screenshot_n = 0

    # Loop principal
    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        fps_counter += 1

        if fps_counter % 30 == 0:
            fps_val     = 30 / max(time.time() - fps_t, 1e-9)
            fps_t       = time.time()
            fps_counter = 0

        # Estimativa do olhar
        feat    = estimator.process(frame)
        face_ok = feat is not None

        if face_ok and model.is_trained:
            ear = feat.get('ear', 0.3)

            # Piscada (natural ou intencional): a íris "salta" e o olhar
            # estimado deixa de ser confiável → congela cursor, dwell e scroll
            gaze_blocked = eye_state.update(ear)

            if not gaze_blocked:
                raw_x, raw_y     = model.predict(feat['features'])
                raw_x, raw_y     = raw_x + bias[0], raw_y + bias[1]
                sx, sy           = smoother.update(raw_x, raw_y)
                gaze_sx, gaze_sy = _to_screen(sx, sy)
                cursor.move(gaze_sx, gaze_sy)
                overlay.move(gaze_sx, gaze_sy)

            # Blink click (intencional, olhos fechados ≥ BLINK_MIN_MS)
            blink_result = None
            if blink_on and blink:
                blink_result = blink.update(ear, gaze_sx, gaze_sy)
                if blink_result:
                    metrics.record_click(gaze_sx, gaze_sy, dwell_ms=0)
                    dwell.reset()          # evita dwell logo após a piscada
                    print(f'  [BLINK] {blink_result}  @ ({gaze_sx}, {gaze_sy})')

            if not gaze_blocked:
                # Dwell click
                if dwell_on and not blink_result:
                    if dwell.update(gaze_sx, gaze_sy):
                        ax, ay = dwell.anchor
                        metrics.record_click(int(ax), int(ay),
                                             dwell_ms=DWELL_TIME_MS)
                        print(f'  [DWELL] ({int(ax)}, {int(ay)})')

                # Edge scroll
                scroller.update(gaze_sy)

        dt_ms = (time.time() - t0) * 1000
        metrics.record_frame(dt_ms)

        # Janela de debug
        if debug_win:
            dbg = frame.copy()

            if face_ok:
                dbg = estimator.draw_debug(dbg, feat)

            # Indicador de dwell
            if dwell_on and face_ok:
                fx = int(gaze_sx * cam_w / SCREEN_W)
                fy = int(gaze_sy * cam_h / SCREEN_H)
                _draw_dwell_arc(dbg, fx, fy, dwell.progress)

            # Indicador de zona de scroll
            _draw_scroll_zone(dbg, cam_h, cam_w, scroller.active_zone)

            # Painel de info
            ear          = feat.get('ear', 0.0) if face_ok else 0.0
            status_face  = 'Detectada' if face_ok else 'Ausente'
            status_calib = 'Sim'       if model.is_trained else 'Nao'
            status_dwell = 'ON'        if dwell_on  else 'OFF'
            status_blink = 'ON'        if blink_on  else 'OFF'
            color_face   = _GREEN if face_ok else _RED

            info = [
                (f'FPS: {fps_val:.1f}  |  Latencia: {dt_ms:.1f} ms', _CYAN),
                (f'Face: {status_face}', color_face),
                (f'Calibrado: {status_calib}', _WHITE),
                (f'Dwell: {status_dwell}  |  Blink: {status_blink}',
                 _GREEN if (dwell_on or blink_on) else _GRAY),
                (f'Gaze (tela): ({gaze_sx}, {gaze_sy})', _YELLOW),
            ]

            if face_ok:
                ear_txt = f'EAR: {ear:.3f}  limiar: {eye_state.threshold:.3f}'
                if eye_state.blocked:
                    ear_txt += '  [PISCADA - olhar ignorado]'
                if blink_on and blink and blink.eye_closed:
                    ear_txt += '  [BLINK]'
                info.append((ear_txt,
                             _ORANGE if eye_state.blocked else _YELLOW))
                info.append((f'Piscadas detectadas: {eye_state.blinks}', _GRAY))

            if dwell_on and face_ok:
                info.append((f'Dwell: {dwell.progress*100:.0f}%', _YELLOW))

            _draw_info_panel(dbg, info)

            cv2.putText(dbg,
                        'Q=Sair  R=Recalib  F=Ajuste  D=Dwell  B=Blink  M=Metricas  S=Screen',
                        (6, cam_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GRAY, 1)

            cv2.imshow(debug_win, dbg)

        # Teclas
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key == ord('r'):
            print('\n  Iniciando recalibração...')
            overlay.hide()
            if _do_recalibration(model, estimator, cap, smoother, dwell,
                                 blink, scroller):
                bias = _fine_tune(model, estimator, cap, smoother)

        elif key == ord('f'):
            print('\n  Refazendo ajuste fino...')
            overlay.hide()
            bias = _fine_tune(model, estimator, cap, smoother, bias)
            dwell.reset()

        elif key == ord('d'):
            dwell_on = not dwell_on
            dwell.reset()
            print(f'  Dwell click: {"ATIVADO" if dwell_on else "DESATIVADO"}')

        elif key == ord('b'):
            if blink is None:
                blink    = BlinkClicker()
                blink_on = True
                print('  Blink click: ATIVADO')
            else:
                blink_on = not blink_on
                blink.reset()
                print(f'  Blink click: {"ATIVADO" if blink_on else "DESATIVADO"}')

        elif key == ord('m'):
            metrics.print_summary()
            metrics.save_csv('eyenav_metrics.csv')

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
    overlay.close()
    estimator.close()
    if blink:
        blink.reset()
    scroller.reset()
    print('  Encerrado. Até a próxima!')


# Funções auxiliares

def _calibrate(model, estimator, cap) -> bool:
    """Executa a tela de calibração, treina e salva o modelo."""
    screen = CalibrationScreen(SCREEN_W, SCREEN_H)
    features, targets = screen.run(estimator, cap)
    if not features:
        return False
    model.train(features, targets)
    model.save(CALIB_FILE)
    print(f'  Calibração concluída com {len(features)} amostras.\n')
    return True


def _run_calibration_if_needed(model, estimator, cap):
    if model.load(CALIB_FILE):
        resp = input('\n  Calibração encontrada. Deseja recalibrar? (s/N): ')
        if resp.strip().lower() != 's':
            return

    print()
    print('  ┌────────────────────────────────────────────┐')
    print('  │  CALIBRAÇÃO                                │')
    print('  │  Mova apenas os olhos, não a cabeça.       │')
    print('  │  Fixe o olhar em cada círculo até          │')
    print('  │  o arco completar.                         │')
    print('  │  ESC = cancelar                            │')
    print('  └────────────────────────────────────────────┘')
    input('  Pressione ENTER para iniciar...')

    if not _calibrate(model, estimator, cap):
        print('  [AVISO] Calibração cancelada ou incompleta.')


def _do_recalibration(model, estimator, cap, smoother, dwell,
                      blink=None, scroller=None) -> bool:
    if _calibrate(model, estimator, cap):
        smoother.reset()
        dwell.reset()
        if blink:
            blink.reset()
        if scroller:
            scroller.reset()
        return True
    print('  Recalibração cancelada.\n')
    return False


def _fine_tune(model, estimator, cap, smoother,
               current_bias=(0.0, 0.0)) -> tuple[float, float]:
    """
    Executa o ajuste fino (triângulo) e aplica o resultado:
    ajusta o filtro e retorna o viés (x, y) a somar às previsões.
    Se cancelado, mantém o viés atual.
    """
    if not (FINE_TUNE_ENABLED and model.is_trained):
        return current_bias

    print('\n  Ajuste fino: fixe o olhar em cada ponto até ele desaparecer.')
    res = FineTuneScreen(SCREEN_W, SCREEN_H).run(model, estimator, cap)
    if res is None:
        print('  Ajuste fino ignorado.\n')
        return current_bias

    smoother.set_min_cutoff(res.min_cutoff)
    smoother.reset()
    print()
    return (res.bias_x, res.bias_y)


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
    print('  ┌────────────────────────────────────────────────────┐')
    print('  │  SISTEMA ATIVO – Controles na janela de debug:     │')
    print('  │   Q / ESC → Encerrar                               │')
    print('  │   R       → Recalibrar                             │')
    print('  │   F       → Refazer ajuste fino (3 pontos)         │')
    print('  │   D       → Ativar / desativar dwell click         │')
    print('  │   B       → Ativar / desativar blink click         │')
    print('  │   M       → Exibir e salvar métricas (CSV)         │')
    print('  │   S       → Salvar screenshot do debug             │')
    print('  └────────────────────────────────────────────────────┘')
    print()


if __name__ == '__main__':
    main()