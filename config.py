# EyeNav – Configurações do sistema
# Ajuste estas constantes conforme seu hardware e preferências.
# Não é necessário alterar o restante do código para customizar o sistema.

# Câmera
WEBCAM_INDEX  = 0       # Índice da câmera (0 = padrão, 1 = segunda câmera)
WEBCAM_WIDTH  = 1280
WEBCAM_HEIGHT = 720
TARGET_FPS    = 30

# Calibração
CALIB_COLS         = 4      # Colunas da grade de calibração
CALIB_ROWS         = 4      # Linhas  (total = COLS × ROWS = 16 pontos)
CALIB_MARGIN       = 0.10   # Margem das bordas (proporção da tela)
CALIB_FACE_WAIT_MS = 2000   # Rosto estável antes de começar
CALIB_PULSE_MS     = 1000   # Pulso: olho chega ao ponto (sem coleta)
CALIB_CAPTURE_MS   = 1000   # Captura: cada frame vira uma amostra
# Ajuste fino (triângulo de 3 pontos, após a calibração)
FINE_TUNE_ENABLED          = True
FINE_TUNE_PULSE_MS         = 1000   # Pulso: olho chega ao ponto (sem coleta)
FINE_TUNE_CAPTURE_MS       = 2000   # Tempo de olhar CONFIRMADO para o ponto sumir
FINE_TUNE_ACCEPT_RADIUS_PX = 200    # Olhar previsto a menos disso do ponto = confirmado
FINE_TUNE_TIMEOUT_MS       = 10000  # Sem confirmar nesse tempo → ajuste abortado
FINE_TUNE_TARGET_JITTER_PX = 12     # Tremor residual desejado com olhar parado
FINE_TUNE_MIN_CUTOFF_FLOOR = 0.15   # Hz. Nunca suavizar mais que isso (cursor travaria)
FINE_TUNE_MAX_BIAS_PX      = 150    # Viés maior que isso não é aplicado (indica calibração ruim)

FEATURE_VERSION  = 2       # Muda quando as features mudam (invalida calibração antiga)
CALIB_FILE       = 'calibration.pkl'

# Landmarks MediaPipe Face Mesh (refine_landmarks=True)
# Íris: 468–472 (esquerda), 473–477 (direita)
LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473

# Cantos e bordas dos olhos
LEFT_EYE_OUTER    = 33
LEFT_EYE_INNER    = 133
LEFT_EYE_TOP      = 159
LEFT_EYE_BOTTOM   = 145

RIGHT_EYE_INNER   = 362
RIGHT_EYE_OUTER   = 263
RIGHT_EYE_TOP     = 386
RIGHT_EYE_BOTTOM  = 374

# Suavização do olhar
SMOOTHER            = 'one_euro'  # 'one_euro' (recomendado) ou 'ema'
MEDIAN_WINDOW       = 7      # Frames da mediana (remove picos isolados)
ONE_EURO_MIN_CUTOFF = 0.15   # Hz. ↓ = menos tremor com o olhar parado
ONE_EURO_BETA       = 0.001  # ↑ = menos atraso quando o olhar se move
ONE_EURO_D_CUTOFF   = 1.0    # Hz. Suavização da estimativa de velocidade
SMOOTH_ALPHA        = 0.12   # Usado apenas se SMOOTHER = 'ema'

# Mapeamento final do cursor
CURSOR_GAIN_X = 1.00   # Expande o movimento horizontal a partir do centro (1.0 = desligado)
CURSOR_GAIN_Y = 1.00   # Expande o movimento vertical (o sinal vertical da íris é mais fraco)
EDGE_SNAP_PX  = 50     # A menos disso da borda, o cursor "gruda" na borda

# Dwell Click
DWELL_TIME_MS   = 1500   # Milissegundos fixando para acionar clique
DWELL_RADIUS_PX = 60     # Raio de tolerância (pixels)
DWELL_REARM_PX  = 150    # Após clicar, distância para permitir novo clique

# Interface
SHOW_DEBUG_WINDOW  = True  # Janela OpenCV com feed da câmera e métricas
DEBUG_WINDOW_SCALE = 0.55  # Escala da janela de debug
CURSOR_MOVE        = True  # False = só debug, sem mover cursor do SO
GAZE_DOT_ENABLED   = True  # Bola sobre a tela no ponto de olhar (X11)
GAZE_DOT_RADIUS    = 12    # Raio da bola (px)
GAZE_DOT_COLOR     = (255, 0, 0)  # RGB

# Supressão de piscadas (sempre ativa)
BLINK_SUPPRESS_RATIO = 0.75  # Olhar ignorado quando EAR < 75% do EAR de olho aberto
BLINK_GRACE_MS       = 200   # Olhar continua ignorado por este tempo após reabrir

# Blink Click
BLINK_CLICK_ENABLED      = False  # Liga com a tecla B durante o uso
BLINK_EAR_THRESHOLD      = 0.12   # Limiar inicial, até aprender o EAR de olho aberto
BLINK_EAR_RATIO          = 0.55   # Fechado = EAR < 55% do EAR de olho aberto
BLINK_MIN_MS             = 2000   # Olhos fechados por 2 s → clique simples
BLINK_MAX_MS             = 4000
BLINK_RIGHT_MIN_MS       = 4000   # Olhos fechados por 4 s → clique direito
BLINK_RIGHT_MAX_MS       = 6000
BLINK_DOUBLE_INTERVAL_MS = 1000   # Pausa máxima entre os dois fechamentos do duplo

# Edge Scroll
SCROLL_ENABLED_DEFAULT   = True   # Estado inicial do scroll pelo olhar
SCROLL_BUTTON_ENABLED    = True   # Botão na tela para ligar/desligar por dwell
SCROLL_BUTTON_SIDE       = 'left' # 'left' ou 'right' (centralizado na vertical)
SCROLL_BUTTON_W          = 90     # Largura (px)
SCROLL_BUTTON_H          = 220    # Altura (px) — alto para tolerar o erro vertical
SCROLL_ZONE_PX           = 80
SCROLL_SPEED             = 3
SCROLL_INTERVAL_MS       = 150