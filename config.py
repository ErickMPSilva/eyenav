# EyeNav – Configurações do sistema
# Ajuste estas constantes conforme seu hardware e preferências.
# Não é necessário alterar o restante do código para customizar o sistema.

# Câmera
WEBCAM_INDEX  = 0       # Índice da câmera (0 = padrão, 1 = segunda câmera)
WEBCAM_WIDTH  = 1280 
WEBCAM_HEIGHT = 720
TARGET_FPS    = 30

# Calibração
CALIB_COLS       = 3       # Colunas da grade de calibração
CALIB_ROWS       = 3       # Linhas  (total = COLS × ROWS = 9 pontos)
CALIB_MARGIN_X   = 0.10   # Margem horizontal (proporção da tela)
CALIB_MARGIN_Y   = 0.10   # Margem vertical
CALIB_SAMPLES    = 40      # Amostras coletadas por ponto de calibração
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

# Suavização EMA
# α ∈ (0, 1):  menor = mais suave (mais lag) | maior = mais responsivo (mais ruído)
SMOOTH_ALPHA = 0.12

# Dwell Click
DWELL_TIME_MS   = 1500   # Milissegundos fixando para acionar clique
DWELL_RADIUS_PX = 60     # Raio de tolerância (pixels)

# Interface
SHOW_DEBUG_WINDOW  = True  # Janela OpenCV com feed da câmera e métricas
DEBUG_WINDOW_SCALE = 0.55  # Escala da janela de debug
CURSOR_MOVE        = True  # False = só debug, sem mover cursor do SO
