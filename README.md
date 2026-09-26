# EyeNav – Navegação Web por Rastreamento Ocular

**TC II – Ciência da Computação | UNIP EAD | 2026**
**Autor:** Erick Martins Paulino Silva
**Linha de pesquisa:** Tecnologia e Bem-Estar Social

---

## Sobre o projeto

Sistema de software que permite a pessoas com imobilidade física navegar
pela internet usando exclusivamente os movimentos dos olhos, via webcam
convencional e visão computacional (Python + MediaPipe + OpenCV).

---

## Pré-requisitos

- Python **3.10** ou superior
- Webcam (integrada ou USB) com ao menos **720p**
- Boa iluminação frontal (evite luz atrás de você)
- Sistema: Windows 10/11, macOS 12+, ou Ubuntu 20.04+

> **Linux:** instale `python3-tk` e `xdotool` para suporte completo ao cursor.
>
> ```bash
> sudo apt install python3-tk xdotool
> ```

---

## Instalação

```bash
# 1. Clone ou extraia o projeto
cd eyenav/

# 2. Crie ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## Execução

```bash
python main.py
```

### Fluxo de primeira execução

1. O sistema abre a webcam e verifica se há calibração salva.
2. Se não houver, inicia a **tela de calibração** automaticamente.
3. Após calibrar, o sistema ativa o rastreamento ocular em tempo real.

---

## Calibração

A calibração mapeia os movimentos da íris para coordenadas da tela.

**Instruções:**

1. Sente-se a ~50–70 cm da câmera.
2. Mantenha a cabeça relativamente parada.
3. Fixe o olhar em cada círculo até o arco completar (~1,5 s).
4. Repita para os 9 pontos da grade (3×3).

A calibração é salva em `calibration.pkl` e recarregada nas próximas sessões.
Pressione **R** na janela de debug para recalibrar.

---

## Controles (janela de debug)

| Tecla   | Ação                                              |
| ------- | ------------------------------------------------- |
| Q / ESC | Encerrar o sistema                                |
| R       | Recalibrar                                        |
| D       | Ativar / desativar dwell click                    |
| M       | Exibir métricas da sessão no console e salvar CSV |
| S       | Salvar screenshot do debug                        |

---

## Dwell Click

O **dwell click** é o mecanismo de seleção: manter o olhar fixo sobre
um ponto por **1,5 segundos** dispara um clique do mouse nessa posição.

- O indicador circular na janela de debug mostra o progresso.
- Qualquer movimento além de **60 px** reinicia o contador.
- Ajuste `DWELL_TIME_MS` e `DWELL_RADIUS_PX` em `config.py`.

---

## Configuração

Edite `config.py` para ajustar:

| Parâmetro         | Padrão | Descrição                       |
| ----------------- | ------ | ------------------------------- |
| `WEBCAM_INDEX`    | 0      | Índice da câmera                |
| `SMOOTH_ALPHA`    | 0.12   | Suavização EMA (↓ = mais suave) |
| `DWELL_TIME_MS`   | 1500   | Tempo para dwell click (ms)     |
| `DWELL_RADIUS_PX` | 60     | Raio de tolerância (px)         |
| `CALIB_COLS/ROWS` | 3×3    | Grade de calibração             |

---

## Estrutura do projeto

```
eyenav/
├── main.py              # Ponto de entrada — loop principal
├── config.py            # Todas as constantes do sistema
├── gaze_estimator.py    # MediaPipe Face Mesh + extração da íris
├── gaze_model.py        # Modelo de calibração (regressão polinomial)
├── calibration.py       # Tela de calibração visual (OpenCV fullscreen)
├── smoother.py          # Suavização EMA do ponto de olhar
├── dwell_clicker.py     # Mecanismo de dwell click
├── cursor_controller.py # Controle do cursor do sistema (pyautogui)
├── metrics.py           # Coleta de métricas de desempenho
├── requirements.txt     # Dependências Python
└── README.md            # Este arquivo
```

---

## Pipeline técnico

```
Webcam frame
    │
    ▼
GazeEstimator (MediaPipe Face Mesh)
  • 478 landmarks faciais
  • Íris esquerda: landmark 468
  • Íris direita:  landmark 473
  • Calcula posição normalizada da íris no olho
    │
    ▼  gaze_vector = (norm_x, norm_y) ∈ [-1, 1]²
    │
    ▼
GazeModel (Regressão Polinomial grau 2 + Ridge)
  • Features: [gx, gy, gx², gy², gx·gy, 1]
  • Prediz: screen_x, screen_y
    │
    ▼  (raw_x, raw_y) em pixels da tela
    │
    ▼
EMASmoother  [s_t = α·x_t + (1-α)·s_{t-1}]
    │
    ▼  (smooth_x, smooth_y)
    │
    ├──► CursorController → pyautogui.moveTo()
    │
    └──► DwellClicker → pyautogui.click() após DWELL_TIME_MS
```

---

## Métricas coletadas (Capítulo 4 do TC II)

Após usar o sistema, pressione **M** ou encerre normalmente.
O arquivo `eyenav_metrics.csv` conterá:

- `timestamp` — momento do evento
- `gaze_x`, `gaze_y` — ponto de olhar estimado
- `target_x`, `target_y` — alvo (se em modo de teste)
- `error_px` — erro euclidiano em pixels
- `dwell_ms` — tempo de dwell acumulado

---

## Dicas para melhores resultados

1. **Iluminação:** luz frontal difusa. Evite luz forte atrás de você.
2. **Distância:** 50–70 cm da câmera.
3. **Calibração:** use 9 pontos (3×3) para melhor precisão.
4. **Suavização:** reduza `SMOOTH_ALPHA` se o cursor tremer muito.
5. **Dwell time:** aumente `DWELL_TIME_MS` se houver cliques acidentais.

---

## Dependências principais

| Biblioteca    | Versão mín. | Uso                               |
| ------------- | ----------- | --------------------------------- |
| mediapipe     | 0.10.0      | Face Mesh + iris landmarks        |
| opencv-python | 4.8.0       | Captura + processamento de imagem |
| numpy         | 1.24.0      | Operações matriciais              |
| scikit-learn  | 1.3.0       | Regressão polinomial (calibração) |
| pyautogui     | 0.9.54      | Controle do cursor e clique       |
| screeninfo    | 0.8.1       | Detecção da resolução da tela     |

---

## Referências bibliográficas

- BRADSKI, G.; KAEHLER, A. _Learning OpenCV_. O'Reilly, 2008.
- KRAFKA, K. et al. Eye tracking for everyone. _IEEE CVPR_, 2016.
- MAJARANTA, P.; BULLING, A. Eye tracking and eye-based HCI. Springer, 2014.
- MUNN, S. M. et al. How do I fixate thee? _ETRA_, 2008.
- SIBERT, J. L.; JACOB, R. J. K. Evaluation of eye gaze interaction. _ACM CHI_, 2000.
- ZHANG, X. et al. Appearance-based gaze estimation in the wild. _IEEE CVPR_, 2015.
- ZHU, Z.; JI, Q. Eye and gaze tracking. _Machine Vision and Applications_, 2005.
