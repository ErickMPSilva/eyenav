"""
Modelo de calibração: gaze vector → coordenadas de tela.

Abordagem: Regressão Polinomial de grau 2 com regularização Ridge,
separada para os eixos X e Y da tela.

Features de entrada  : [lnx, lny, rnx, rny, l_lid, r_lid]  (ver gaze_estimator)
Pré-processamento    : padronização (StandardScaler) — média 0, desvio 1
Features expandidas  : polinômio de grau 2 (termos, quadrados e produtos)
Targets              : screen_x  /  screen_y

A padronização é necessária porque o vetor de olhar varia pouco
(tipicamente ±0,1 a ±0,2). Sem ela, a penalidade L2 do Ridge atua
sobre coeficientes da ordem de milhares e "encolhe" as previsões
para o centro da tela.

Esta abordagem é standard na literatura de low-cost eye tracking
(Munn et al., 2008; Zhu & Ji, 2005) e lida melhor com não-linearidades
do que a simples regressão linear.

Serialização via pickle permite salvar e reutilizar a calibração
entre sessões sem precisar recalibrar toda vez.
"""

import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from config import CALIB_FILE, FEATURE_VERSION


class GazeModel:
    """
    Modelo de calibração baseado em regressão polinomial.

    Treina dois pipelines independentes:
      - model_x : gaze_vector → screen_x
      - model_y : gaze_vector → screen_y
    """

    def __init__(self, degree: int = 2, alpha: float = 1.0):
        """
        Parameters
        ----------
        degree : int
            Grau do polinômio (2 é suficiente para a maioria dos usuários).
        alpha  : float
            Regularização L2 (Ridge). Aumentar se houver overfitting.
        """
        self._degree    = degree
        self._alpha     = alpha
        self._model_x   = self._make_pipeline()
        self._model_y   = self._make_pipeline()
        self.is_trained = False

    # API pública

    def train(self, features: list[tuple], targets: list[tuple]):
        """
        Treina o modelo com os dados coletados na calibração.

        Parameters
        ----------
        features : list of (gaze_x, gaze_y)
            Vetores de olhar — um por ponto de calibração.
        targets  : list of (screen_x, screen_y)
            Coordenadas de tela correspondentes.

        Raises
        ------
        ValueError se os dados forem insuficientes (< 6 pontos).
        """
        if len(features) < 6:
            raise ValueError('São necessários ao menos 6 pontos para calibração.')

        X = np.array(features, dtype=float)   # (n, 2)
        y = np.array(targets,  dtype=float)   # (n, 2)

        self._model_x.fit(X, y[:, 0])
        self._model_y.fit(X, y[:, 1])
        self.is_trained = True

        # Diagnóstico: erro nos próprios pontos de calibração
        pred_x = self._model_x.predict(X)
        pred_y = self._model_y.predict(X)
        erros  = np.hypot(pred_x - y[:, 0], pred_y - y[:, 1])
        print(f'[GazeModel] Erro nos pontos de calibração: '
              f'média {erros.mean():.0f} px | máx {erros.max():.0f} px')
        print(f'[GazeModel] Faixa da íris (olho esq.): '
              f'x [{X[:, 0].min():+.3f}, {X[:, 0].max():+.3f}]  '
              f'y [{X[:, 1].min():+.3f}, {X[:, 1].max():+.3f}]')
        if X.shape[1] >= 6:
            print(f'[GazeModel] Faixa da pálpebra (esq.): '
                  f'[{X[:, 4].min():+.3f}, {X[:, 4].max():+.3f}]')
        print(f'[GazeModel] Faixa prevista na tela: '
              f'x [{pred_x.min():.0f}, {pred_x.max():.0f}]  '
              f'y [{pred_y.min():.0f}, {pred_y.max():.0f}]')

    def predict(self, features) -> tuple[int, int]:
        """
        Prediz a posição na tela a partir do vetor de olhar.

        Returns
        -------
        (screen_x, screen_y) : tuple of int
        """
        if not self.is_trained:
            raise RuntimeError('Modelo não treinado. Execute a calibração.')

        X = np.asarray(features, dtype=float).reshape(1, -1)
        sx = int(round(self._model_x.predict(X)[0]))
        sy = int(round(self._model_y.predict(X)[0]))
        return sx, sy

    def save(self, path: str = CALIB_FILE):
        """Salva o modelo treinado em disco (pickle)."""
        data = {
            'model_x':  self._model_x,
            'model_y':  self._model_y,
            'degree':   self._degree,
            'trained':  self.is_trained,
            'version':  FEATURE_VERSION,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f'[GazeModel] Calibração salva → {path}')

    def load(self, path: str = CALIB_FILE) -> bool:
        """
        Carrega um modelo previamente salvo.

        Returns
        -------
        True se bem-sucedido, False se arquivo não encontrado.
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if data.get('version') != FEATURE_VERSION:
                print('[GazeModel] Calibração salva usa features antigas — '
                      'será necessário recalibrar.')
                return False
            self._model_x   = data['model_x']
            self._model_y   = data['model_y']
            self._degree    = data['degree']
            self.is_trained = data['trained']
            print(f'[GazeModel] Calibração carregada ← {path}')
            return True
        except FileNotFoundError:
            return False

    # Privado

    def _make_pipeline(self) -> Pipeline:
        return Pipeline([
            ('scale', StandardScaler()),
            ('poly',  PolynomialFeatures(degree=self._degree, include_bias=False)),
            ('ridge', Ridge(alpha=self._alpha)),
        ])