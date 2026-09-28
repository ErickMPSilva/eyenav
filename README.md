# EyeNav – Navegação Web por Rastreamento Ocular

**TC II – Ciência da Computação | UNIP EAD | 2026**
**Autor:** Erick Martins Paulino Silva
**Linha de pesquisa:** Tecnologia e Bem-Estar Social

---

## Sobre o projeto

Sistema de software que permite a pessoas com imobilidade física navegar
pela internet usando exclusivamente os movimentos dos olhos, via webcam
convencional e visão computacional (Python + MediaPipe + OpenCV).

O usuário move o cursor com o olhar e interage com a página por três
mecanismos sem uso das mãos:

- **Dwell click:** fixar o olhar em um ponto para clicar.
- **Blink click:** piscar intencionalmente para clique simples, duplo ou direito.
- **Edge scroll:** olhar para a borda superior ou inferior da tela para rolar a página.

---

## Pré-requisitos

- Python **3.10** ou superior
- Webcam (integrada ou USB), preferencialmente **720p**
- Boa iluminação frontal (evite luz atrás de você)
- Sistema: Ubuntu 20.04+ (X11 ou Wayland), Windows 10/11 ou macOS 12+

### Backend de controle do mouse

O EyeNav precisa de um backend para mover o cursor e clicar. O módulo
`mouse_backend.py` detecta automaticamente o melhor disponível, nesta ordem:

| Prioridade | Backend   | Ambiente            | Instalação                                     |
| ---------- | --------- | ------------------- | ---------------------------------------------- |
| 1          | `ydotool` | Linux (Wayland)     | `sudo apt install ydotool` + daemon `ydotoold` |
| 2          | `xdotool` | Linux (X11)         | `sudo apt install xdotool`                     |
| 3          | `pynput`  | X11, Windows, macOS | instalado via `requirements.txt`               |

Ao iniciar, o sistema informa o backend escolhido:

```
[MouseBackend] Sessão: x11 (X11/outra) | Backend: xdotool
```

> **Importante:** se aparecer `Backend: none`, os cliques são detectados
> (aparecem no console e no CSV), mas **não são executados** no sistema.
> Instale um dos backends acima.

No Linux, recomenda-se também o `python3-tk`:

```bash
sudo apt install python3-tk xdotool
```

No macOS, conceda permissão de **Acessibilidade** ao terminal em
_Ajustes do Sistema → Privacidade e Segurança_, necessária para o `pynput`
controlar o mouse.

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

Na primeira execução, o modelo `face_landmarker.task` (~6 MB) é baixado
automaticamente caso a versão instalada do MediaPipe use a Tasks API.

---

## Execução

```bash
python main.py
```

### Fluxo de primeira execução

1. O sistema detecta o backend de mouse e abre a webcam.
2. Verifica se há calibração salva (`calibration.pkl`).
3. Se não houver, inicia a **tela de calibração** automaticamente.
4. Após calibrar, o rastreamento ocular é ativado em tempo real.

---

## Calibração

A calibração mapeia a posição da íris para coordenadas da tela.

**Instruções:**

1. Sente-se a ~50–70 cm da câmera.
2. Mantenha a cabeça relativamente parada.
3. Fixe o olhar em cada círculo até o arco completar.
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
| B       | Ativar / desativar blink click                    |
| M       | Exibir métricas da sessão no console e salvar CSV |
| S       | Salvar screenshot do debug                        |

---

## Mecanismos de interação

### Dwell click

Manter o olhar fixo sobre um ponto por **1,5 segundo** dispara um clique
esquerdo nessa posição.

- O indicador circular na janela de debug mostra o progresso.
- Qualquer movimento além de **60 px** reinicia o contador.
- Ajuste `DWELL_TIME_MS` e `DWELL_RADIUS_PX` em `config.py`.

### Blink click

Piscadas intencionais são detectadas pelo **EAR** (_Eye Aspect Ratio_,
razão altura/largura do olho). O olho é considerado fechado quando o EAR
fica abaixo de `BLINK_EAR_THRESHOLD`.

| Piscada                                          | Duração (padrão) | Ação           |
| ------------------------------------------------ | ---------------- | -------------- |
| Involuntária                                     | < 500 ms         | ignorada       |
| Curta                                            | 500–800 ms       | clique simples |
| Duas curtas dentro de `BLINK_DOUBLE_INTERVAL_MS` | —                | duplo clique   |
| Longa                                            | 800–1500 ms      | clique direito |
| Excessiva                                        | > 1500 ms        | ignorada       |

O valor do EAR aparece na janela de debug. Observe-o com o olho aberto e
fechado para ajustar o limiar ao seu rosto e iluminação.

### Edge scroll

Olhar para a faixa de **80 px** no topo ou na base da tela rola a página
para cima ou para baixo. O scroll começa após 300 ms na zona (evita
acionamento acidental) e continua enquanto o olhar permanecer nela.

---

## Configuração

Todas as constantes ficam em `config.py`.

| Parâmetro                   | Padrão   | Descrição                                       |
| --------------------------- | -------- | ----------------------------------------------- |
| `WEBCAM_INDEX`              | 1        | Índice da câmera (0 = câmera padrão)            |
| `WEBCAM_WIDTH/HEIGHT`       | 1280×720 | Resolução solicitada à câmera                   |
| `SMOOTH_ALPHA`              | 0.12     | Suavização EMA (↓ = mais suave, mais lag)       |
| `CALIB_COLS/ROWS`           | 3×3      | Grade de calibração                             |
| `CALIB_SAMPLES`             | 40       | Amostras coletadas por ponto                    |
| `DWELL_TIME_MS`             | 1500     | Tempo de fixação para dwell click (ms)          |
| `DWELL_RADIUS_PX`           | 60       | Raio de tolerância do dwell (px)                |
| `BLINK_CLICK_ENABLED`       | True     | Ativa o blink click ao iniciar                  |
| `BLINK_EAR_THRESHOLD`       | 0.12     | EAR abaixo do qual o olho é considerado fechado |
| `BLINK_MIN_MS/MAX_MS`       | 500/800  | Faixa da piscada curta (clique simples)         |
| `BLINK_RIGHT_MIN_MS/MAX_MS` | 800/1500 | Faixa da piscada longa (clique direito)         |
| `BLINK_DOUBLE_INTERVAL_MS`  | 400      | Janela para o duplo clique (ms)                 |
| `SCROLL_ZONE_PX`            | 80       | Altura das zonas de scroll (px)                 |
| `SCROLL_SPEED`              | 3        | Passos de rolagem por tick                      |
| `SCROLL_INTERVAL_MS`        | 150      | Intervalo entre ticks de scroll (ms)            |
| `CURSOR_MOVE`               | True     | False = apenas debug, sem mover o cursor        |
| `SHOW_DEBUG_WINDOW`         | True     | Exibe a janela de debug                         |

---

## Estrutura do projeto

```
eyenav/
├── main.py              # Ponto de entrada — loop principal
├── config.py            # Todas as constantes do sistema
├── gaze_estimator.py    # MediaPipe Face Mesh + extração da íris e EAR
├── gaze_model.py        # Modelo de calibração (regressão polinomial)
├── calibration.py       # Tela de calibração visual (OpenCV fullscreen)
├── smoother.py          # Suavização EMA do ponto de olhar
├── mouse_backend.py     # Backend unificado de mouse (xdotool/ydotool/pynput)
├── cursor_controller.py # Movimento do cursor do sistema
├── dwell_clicker.py     # Clique por fixação do olhar
├── blink_clicker.py     # Clique por piscada intencional
├── edge_scroller.py     # Scroll pelas bordas da tela
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
  • Íris esquerda: landmark 468 | Íris direita: landmark 473
  • Posição normalizada da íris no olho + EAR
    │
    ▼  gaze_vector = (norm_x, norm_y) ∈ [-1, 1]²
    │
    ▼
GazeModel (Regressão Polinomial grau 2 + Ridge)
  • Features: [1, gx, gy, gx², gx·gy, gy²]
  • Prediz: screen_x, screen_y
    │
    ▼  (raw_x, raw_y) em pixels da tela
    │
    ▼
EMASmoother  [s_t = α·x_t + (1-α)·s_{t-1}]
    │
    ▼  (smooth_x, smooth_y)
    │
    ├──► CursorController ──┐
    ├──► DwellClicker ──────┤
    ├──► BlinkClicker (EAR) ┼──► mouse_backend ──► xdotool / ydotool / pynput
    └──► EdgeScroller ──────┘
```

Os módulos de interação decidem **quando** agir. O `mouse_backend` decide
**como** executar a ação no sistema operacional. Essa separação isola a
lógica de interação das particularidades de cada plataforma.

---

## Métricas coletadas (Capítulo 4 do TC II)

Ao encerrar o sistema (ou pressionar **M**), é exibido um resumo com duração,
frames processados, FPS médio, latência média por frame e total de cliques.
O arquivo `eyenav_metrics.csv` contém um registro por clique:

- `timestamp` — momento do evento (Unix time)
- `gaze_x`, `gaze_y` — ponto de olhar estimado
- `target_x`, `target_y` — alvo conhecido (apenas em modo de teste)
- `error_px` — erro euclidiano em pixels (`nan` quando não há alvo)
- `dwell_ms` — tempo de dwell (0 para cliques por piscada)

> O erro médio só é significativo em sessões de teste com alvos conhecidos.
> Em uso livre, `error_px` fica `nan` e o resumo não reflete precisão.

---

## Solução de problemas

| Sintoma                                          | Causa provável / solução                                       |
| ------------------------------------------------ | -------------------------------------------------------------- |
| `Backend: none`, cursor não se move, sem cliques | Nenhum backend instalado. Veja _Backend de controle do mouse_. |
| `can't open camera by index`                     | Ajuste `WEBCAM_INDEX` em `config.py` (tente 0).                |
| Câmera abre em 640×480                           | A câmera não suporta 720p no índice escolhido.                 |
| Cursor treme muito                               | Reduza `SMOOTH_ALPHA` ou melhore a iluminação.                 |
| Cliques acidentais                               | Aumente `DWELL_TIME_MS` ou desative o dwell (tecla D).         |
| Blink click nunca dispara                        | Ajuste `BLINK_EAR_THRESHOLD` observando o EAR no debug.        |
| Avisos `QFontDatabase` no terminal               | Inofensivos (fontes do Qt do OpenCV); podem ser ignorados.     |
| Wayland com `ydotool` não responde               | Verifique se o daemon `ydotoold` está em execução.             |

---

## Dicas para melhores resultados

1. **Iluminação:** luz frontal difusa. Evite luz forte atrás de você.
2. **Distância:** 50–70 cm da câmera.
3. **Calibração:** recalibre (tecla R) se mudar de posição ou de iluminação.
4. **Suavização:** reduza `SMOOTH_ALPHA` se o cursor tremer muito.
5. **Dwell time:** aumente `DWELL_TIME_MS` se houver cliques acidentais.

---

## Dependências principais

| Biblioteca    | Versão mín. | Uso                                           |
| ------------- | ----------- | --------------------------------------------- |
| mediapipe     | 0.10.0      | Face Mesh + landmarks da íris                 |
| opencv-python | 4.8.0       | Captura, processamento e janelas de interface |
| numpy         | 1.24.0      | Operações matriciais                          |
| scikit-learn  | 1.3.0       | Regressão polinomial (calibração)             |
| pynput        | 1.7.6       | Controle do mouse (backend multiplataforma)   |
| screeninfo    | 0.8.1       | Detecção da resolução da tela                 |

Ferramentas de sistema opcionais (Linux): `xdotool` (X11) e `ydotool` (Wayland).

---

## Referências bibliográficas

- BRADSKI, G.; KAEHLER, A. _Learning OpenCV_. O'Reilly, 2008.
- KRAFKA, K. et al. Eye tracking for everyone. _IEEE CVPR_, 2016.
- MAJARANTA, P.; BULLING, A. Eye tracking and eye-based HCI. Springer, 2014.
- MUNN, S. M. et al. How do I fixate thee? _ETRA_, 2008.
- SIBERT, J. L.; JACOB, R. J. K. Evaluation of eye gaze interaction. _ACM CHI_, 2000.
- SOUKUPOVÁ, T.; ČECH, J. Real-time eye blink detection using facial landmarks. _Computer Vision Winter Workshop_, 2016.
- ZHANG, X. et al. Appearance-based gaze estimation in the wild. _IEEE CVPR_, 2015.
- ZHU, Z.; JI, Q. Eye and gaze tracking. _Machine Vision and Applications_, 2005.
