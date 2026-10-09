"""Abordagem 1: ML clássico. TF-IDF de palavras e de pedaços de palavra + regressão logística.

É a linha de base que todo projeto de classificação deveria ter antes de chamar um LLM:
treina em segundos na CPU, responde em frações de milissegundo e não custa nada por chamada.
Os n-gramas de caracteres ajudam com erros de digitação e variações ("cartao", "cartão").
"""

from __future__ import annotations

import pickle
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .dados import Exemplo


def treinar(treino: list[Exemplo], c: float = 10.0) -> Pipeline:
    modelo = Pipeline([
        ("atributos", FeatureUnion([
            ("palavras", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, strip_accents="unicode")),
            ("pedacos", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
        ])),
        ("classificador", LogisticRegression(C=c, max_iter=3000)),
    ])
    modelo.fit([e.frase for e in treino], [e.rotulo for e in treino])
    return modelo


def rodar(treino: list[Exemplo], teste: list[Exemplo]) -> tuple[list[str], dict]:
    inicio = time.perf_counter()
    modelo = treinar(treino)
    segundos_treino = time.perf_counter() - inicio

    inicio = time.perf_counter()
    previsoes = [str(p) for p in modelo.predict([e.frase for e in teste])]
    ms = (time.perf_counter() - inicio) * 1000 / len(teste)
    extras = {
        "parametros": "TF-IDF + regressão logística",
        "segundos_treino": round(segundos_treino, 1),
        "ms_por_exemplo": round(ms, 3),
        "tamanho_mb": round(len(pickle.dumps(modelo)) / 1e6, 1),
        "custo_por_1000": 0.0,
    }
    return previsoes, extras
