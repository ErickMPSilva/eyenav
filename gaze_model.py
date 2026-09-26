"""
Modelo de calibração: gaze vector → coordenadas de tela.

Abordagem: Regressão Polinomial de grau 2 com regularização Ridge,
separada para os eixos X e Y da tela.

Features de entrada  : [gaze_x, gaze_y]
Features expandidas  : [1, gx, gy, gx², gy², gx·gy]
Targets              : screen_x  /  screen_y

Esta abordagem é standard na literatura de low-cost eye tracking
(Munn et al., 2008; Zhu & Ji, 2005) e lida melhor com não-linearidades
do que a simples regressão linear.

Serialização via pickle permite salvar e reutilizar a calibração
entre sessões sem precisar recalibrar toda vez.
"""

import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from config import CALIB_FILE


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

    def predict(self, gaze_vector: tuple) -> tuple[int, int]:
        """
        Prediz a posição na tela a partir do vetor de olhar.

        Parameters
        ----------
        gaze_vector : (float, float)
            Posição normalizada da íris.

        Returns
        -------
        (screen_x, screen_y) : tuple of int
        """
        if not self.is_trained:
            raise RuntimeError('Modelo não treinado. Execute a calibração.')

        X = np.array([[gaze_vector[0], gaze_vector[1]]], dtype=float)
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
            ('poly',  PolynomialFeatures(degree=self._degree, include_bias=True)),
            ('ridge', Ridge(alpha=self._alpha)),
        ])
