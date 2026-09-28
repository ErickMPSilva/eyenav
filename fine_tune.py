"""
Fine tuning pós-calibração (triângulo de 3 pontos).

Executado logo após a calibração — ou após carregar uma calibração
salva. O usuário fixa o olhar em 3 pontos dispostos em triângulo.

Confirmação do olhar:
  Cada ponto só encolhe enquanto a previsão do modelo (mediana dos
  últimos frames) estiver dentro de FINE_TUNE_ACCEPT_RADIUS_PX dele.
  O ponto fica verde quando o olhar é confirmado e vermelho quando não
  é; olhando para outro lugar, o ponto não avança. Só os frames com
  olhar confirmado são usados. Se um ponto não for confirmado em
  FINE_TUNE_TIMEOUT_MS, o ajuste é abortado: a calibração não é boa o
  bastante nessa região e deve ser refeita.

Com os frames confirmados, o sistema mede duas coisas:

1. Viés (offset)
   Diferença média entre o alvo e a previsão. Corrige deslocamentos
   sistemáticos — por exemplo, a cabeça em posição um pouco diferente
   da usada na calibração salva. O viés é somado a toda previsão.

2. Ruído (tremor)
   Com o olhar parado, toda variação da previsão é ruído. As sequências
   gravadas são reprocessadas pelo mesmo pipeline do sistema (mediana +
   One Euro) para vários valores de ONE_EURO_MIN_CUTOFF, e é escolhido
   o MAIOR corte (menor atraso) cujo tremor residual fique abaixo de
   FINE_TUNE_TARGET_JITTER_PX. Assim o filtro é ajustado ao ruído real
   do usuário, da câmera e da iluminação naquele momento, em vez de
   usar um valor fixo.

Ideia inspirada na etapa de ajuste do filtro do EyeTrax (Zhang, 2025),
que mede o ruído durante a fixação de pontos conhecidos. Aqui a mesma
ideia é aplicada ao filtro One Euro, que em simulação apresentou menor
atraso e sem ultrapassagem (overshoot) em relação a um Kalman de
velocidade constante.
"""

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from config import (FINE_TUNE_PULSE_MS, FINE_TUNE_CAPTURE_MS,
                    FINE_TUNE_ACCEPT_RADIUS_PX, FINE_TUNE_TIMEOUT_MS,
                    FINE_TUNE_TARGET_JITTER_PX, FINE_TUNE_MIN_CUTOFF_FLOOR,
                    FINE_TUNE_MAX_BIAS_PX, MEDIAN_WINDOW,
                    BLINK_SUPPRESS_RATIO)
from smoother import OneEuroSmoother

_GREEN = (0, 255, 0)
_RED   = (0, 0, 255)
_TIMEOUT = 'timeout'
_GRAY  = (90, 90, 90)
_WARMUP = 8                                   # amostras descartadas no início
_CUTOFFS = np.geomspace(FINE_TUNE_MIN_CUTOFF_FLOOR, 3.0, 30)   # candidatos (Hz)


@dataclass
class FineTuneResult:
    bias_x:      float     # somar à previsão x (px)
    bias_y:      float     # somar à previsão y (px)
    min_cutoff:  float     # Hz, para o One Euro
    raw_std:     float     # ruído bruto da previsão (px)
    tremor:      float     # tremor residual esperado após o filtro (px)
    samples:     int


def triangle_points(sw: int, sh: int) -> list[tuple[int, int]]:
    """Topo-centro, base-esquerda, base-direita."""
    return [(sw // 2, int(sh * 0.15)),
            (int(sw * 0.15), int(sh * 0.85)),
            (int(sw * 0.85), int(sh * 0.85))]


class FineTuneScreen:
    WIN = 'EyeNav - Ajuste fino'

    def __init__(self, sw: int, sh: int):
        self.sw, self.sh = sw, sh
        self.points = triangle_points(sw, sh)

    # API

    def run(self, model, estimator, cap) -> FineTuneResult | None:
        """Executa o ajuste. Retorna None se cancelado ou sem dados."""
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.WIN, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        recordings = []   # por ponto: (alvo, [(t, px, py, ear), ...])
        for idx, pt in enumerate(self.points):
            if not self._pulse(cap, pt, idx):
                cv2.destroyWindow(self.WIN)
                return None
            rec = self._capture(model, estimator, cap, pt, idx)
            if rec is None:
                cv2.destroyWindow(self.WIN)
                return None
            if rec is _TIMEOUT:
                cv2.destroyWindow(self.WIN)
                print(f'[FineTune] Olhar não confirmado no ponto {idx + 1} '
                      f'{pt} em {FINE_TUNE_TIMEOUT_MS / 1000:.0f} s. '
                      f'A calibração erra demais nessa região — recalibre (R).')
                return None
            recordings.append((pt, rec))

        cv2.destroyWindow(self.WIN)
        return self._analyze(recordings)

    # Etapas visuais

    def _pulse(self, cap, pt, idx) -> bool:
        t0 = time.time()
        while (e := time.time() - t0) * 1000 < FINE_TUNE_PULSE_MS:
            cap.read()
            canvas = self._blank()
            r = 15 + int(15 * abs(np.sin(2 * np.pi * e)))
            cv2.circle(canvas, pt, r, _GREEN, -1)
            self._hud(canvas, idx)
            cv2.imshow(self.WIN, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                return False
        return True

    def _capture(self, model, estimator, cap, pt, idx):
        """
        O ponto só encolhe enquanto o olhar previsto está sobre ele.
        Grava (t, pred_x, pred_y, ear) apenas dos frames confirmados.
        """
        rec     = []
        recent  = deque(maxlen=max(1, MEDIAN_WINDOW))
        done_ms = 0.0
        t_start = t_last = time.time()

        while done_ms < FINE_TUNE_CAPTURE_MS:
            now = time.time()
            dt_ms, t_last = (now - t_last) * 1000, now
            if (now - t_start) * 1000 > FINE_TUNE_TIMEOUT_MS:
                return _TIMEOUT

            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)

            on_target = False
            feat = estimator.process(frame)
            if feat is not None:
                px, py = model.predict(feat['features'])
                recent.append((px, py))
                mx, my = np.median(np.array(recent), axis=0)
                on_target = (np.hypot(mx - pt[0], my - pt[1])
                             <= FINE_TUNE_ACCEPT_RADIUS_PX)
                if on_target:
                    rec.append((now, float(px), float(py), feat['ear']))
            else:
                recent.clear()

            if on_target:
                done_ms += dt_ms

            # Desenho: encolhe conforme o progresso; verde = olhar confirmado
            canvas = self._blank()
            frac   = 1 - done_ms / FINE_TUNE_CAPTURE_MS
            color  = _GREEN if on_target else _RED
            cv2.circle(canvas, pt, max(3, int(25 * frac)), color, -1)
            if not on_target:
                cv2.circle(canvas, pt, 40, _RED, 1)
                self._center_text(canvas, 'Olhe para o ponto')
            self._hud(canvas, idx)
            cv2.imshow(self.WIN, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                return None
        return rec

    # Análise

    def _analyze(self, recordings) -> FineTuneResult | None:
        # Remove frames de piscada (EAR abaixo do limiar relativo à mediana)
        all_ears = [r[3] for _, rec in recordings for r in rec]
        if not all_ears:
            print('[FineTune] Nenhuma amostra válida — ajuste ignorado.')
            return None
        ear_min = np.median(all_ears) * BLINK_SUPPRESS_RATIO

        seqs = []   # por ponto: (alvo, array (n, 3) com t, x, y)
        for pt, rec in recordings:
            arr = np.array([(t, x, y) for t, x, y, ear in rec if ear >= ear_min])
            if len(arr) > _WARMUP + 5:
                seqs.append((pt, arr))

        if not seqs:
            print('[FineTune] Amostras insuficientes — ajuste ignorado.')
            return None

        # 1. Viés: alvo − previsão média (ignora o aquecimento)
        offs = np.array([np.array(pt) - arr[_WARMUP:, 1:].mean(axis=0)
                         for pt, arr in seqs])
        bias_x, bias_y = offs.mean(axis=0)

        # Ruído bruto: desvio em torno da média de cada ponto
        raw_std = float(np.mean([np.linalg.norm(arr[_WARMUP:, 1:].std(axis=0))
                                 for _, arr in seqs]))

        # 2. Escolha do corte: maior corte com tremor ≤ alvo
        best_cut, best_trem = _CUTOFFS[0], self._tremor(seqs, _CUTOFFS[0])
        for cut in _CUTOFFS:
            trem = self._tremor(seqs, cut)
            if trem <= FINE_TUNE_TARGET_JITTER_PX:
                best_cut, best_trem = cut, trem

        n = int(sum(len(a) for _, a in seqs))
        res = FineTuneResult(float(bias_x), float(bias_y), float(best_cut),
                             raw_std, float(best_trem), n)

        print(f'[FineTune] {n} amostras | viés: ({res.bias_x:+.0f}, '
              f'{res.bias_y:+.0f}) px | ruído bruto: {raw_std:.0f} px')
        print(f'[FineTune] min_cutoff = {res.min_cutoff:.3f} Hz → tremor '
              f'residual ≈ {res.tremor:.1f} px '
              f'(alvo ≤ {FINE_TUNE_TARGET_JITTER_PX} px)')
        if res.tremor > FINE_TUNE_TARGET_JITTER_PX:
            print(f'[FineTune] AVISO: tremor acima do alvo mesmo com a '
                  f'suavização máxima permitida ({FINE_TUNE_MIN_CUTOFF_FLOOR} Hz). '
                  f'O ruído vem do sinal, não do filtro.')
        if np.hypot(res.bias_x, res.bias_y) > FINE_TUNE_MAX_BIAS_PX:
            print(f'[FineTune] AVISO: viés de {np.hypot(res.bias_x, res.bias_y):.0f} px '
                  f'é grande demais para ser deslocamento de cabeça — '
                  f'NÃO aplicado. Recalibre (tecla R).')
            res.bias_x = res.bias_y = 0.0
        return res

    @staticmethod
    def _tremor(seqs, min_cutoff: float) -> float:
        """Reprocessa as sequências (mediana + One Euro) e mede o tremor."""
        vals = []
        for _, arr in seqs:
            filt = OneEuroSmoother(min_cutoff=min_cutoff)
            buf  = deque(maxlen=max(1, MEDIAN_WINDOW))
            out  = []
            for t, x, y in arr:
                buf.append((x, y))
                mx, my = np.median(np.array(buf), axis=0)
                out.append(filt.update(mx, my, t=t))
            out = np.array(out[_WARMUP:])
            vals.append(np.linalg.norm(out.std(axis=0)))
        return float(np.mean(vals))

    # Desenho

    def _blank(self):
        return np.zeros((self.sh, self.sw, 3), dtype=np.uint8)

    def _center_text(self, canvas, txt):
        size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        cv2.putText(canvas, txt, ((self.sw - size[0]) // 2, self.sh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, _GRAY, 1)

    def _hud(self, canvas, idx):
        cv2.putText(canvas, f'Ajuste fino {idx + 1}/3', (20, self.sh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GRAY, 1)
        cv2.putText(canvas, 'ESC = pular', (self.sw - 130, self.sh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GRAY, 1)