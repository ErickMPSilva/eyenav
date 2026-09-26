"""
Coleta de métricas de desempenho para o Capítulo 4 do TC II.

Registra:
  - Latência de processamento por frame (ms)
  - Erro de predição do olhar em pontos de teste (px)
  - Eventos de dwell click (coordenadas, tempo acumulado)
  - FPS médio da sessão

Gera CSV ao final para análise estatística (Excel / Python / R).

Uso:
    m = MetricsCollector()
    m.record_frame(dt_ms=16.7)
    m.record_click(gx=940, gy=540, target_x=960, target_y=540, dwell_ms=1500)
    m.print_summary()
    m.save_csv('resultados_sessao1.csv')
"""

import csv
import time
import statistics
from dataclasses import dataclass, field, asdict


@dataclass
class ClickEvent:
    timestamp:  float
    gaze_x:     int
    gaze_y:     int
    target_x:   int | None
    target_y:   int | None
    error_px:   float
    dwell_ms:   float


class MetricsCollector:
    """Coleta e resume métricas de desempenho do sistema."""

    def __init__(self):
        self._frame_times_ms: list[float] = []
        self._clicks:         list[ClickEvent] = []
        self._session_start = time.time()

    # Registro

    def record_frame(self, dt_ms: float):
        """Registra tempo de processamento de um frame."""
        self._frame_times_ms.append(dt_ms)

    def record_click(self, gx: int, gy: int,
                     dwell_ms: float,
                     target_x: int | None = None,
                     target_y: int | None = None):
        """
        Registra evento de dwell click.

        Se target_x/y fornecidos, calcula erro em pixels.
        Use durante testes com alvo conhecido (avaliação de precisão).
        """
        if target_x is not None and target_y is not None:
            err = ((gx - target_x) ** 2 + (gy - target_y) ** 2) ** 0.5
        else:
            err = float('nan')

        self._clicks.append(ClickEvent(
            timestamp=time.time(),
            gaze_x=gx, gaze_y=gy,
            target_x=target_x, target_y=target_y,
            error_px=err, dwell_ms=dwell_ms
        ))

    # Resumo

    def summary(self) -> dict:
        """Retorna dicionário com estatísticas da sessão."""
        ft = self._frame_times_ms
        errs = [c.error_px for c in self._clicks
                if not (c.error_px != c.error_px)]  # exclui NaN

        def safe_mean(lst):  return statistics.mean(lst) if lst else 0.0
        def safe_stdev(lst): return statistics.stdev(lst) if len(lst) > 1 else 0.0

        return {
            'duracao_sessao_s':    round(time.time() - self._session_start, 1),
            'frames_processados':  len(ft),
            'latencia_media_ms':   round(safe_mean(ft),  2),
            'latencia_stdev_ms':   round(safe_stdev(ft), 2),
            'fps_medio':           round(1000 / safe_mean(ft), 1) if ft else 0,
            'total_clicks':        len(self._clicks),
            'erro_medio_px':       round(safe_mean(errs),  1),
            'erro_stdev_px':       round(safe_stdev(errs), 1),
        }

    def print_summary(self):
        """Exibe resumo formatado no console."""
        s = self.summary()
        print()
        print('═' * 50)
        print('  EyeNav – Métricas da Sessão')
        print('═' * 50)
        print(f'  Duração               : {s["duracao_sessao_s"]} s')
        print(f'  Frames processados    : {s["frames_processados"]}')
        print(f'  FPS médio             : {s["fps_medio"]}')
        print(f'  Latência média        : {s["latencia_media_ms"]} ms '
              f'(±{s["latencia_stdev_ms"]} ms)')
        print(f'  Total de clicks       : {s["total_clicks"]}')
        if s['total_clicks'] > 0:
            print(f'  Erro médio (gaze→alvo): {s["erro_medio_px"]} px '
                  f'(±{s["erro_stdev_px"]} px)')
        print('═' * 50)
        print()

    def save_csv(self, path: str = 'eyenav_metrics.csv'):
        """Salva eventos de click em CSV para análise estatística."""
        if not self._clicks:
            print('[Metrics] Nenhum click registrado para salvar.')
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self._clicks[0]).keys())
            writer.writeheader()
            for c in self._clicks:
                writer.writerow(asdict(c))
        print(f'[Metrics] CSV salvo → {path}')
