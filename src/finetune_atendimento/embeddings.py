"""Abordagem 2: embeddings multilíngues prontos + regressão logística.

Um modelo de embeddings (multilingual-e5-small, 118 milhões de parâmetros) transforma cada
frase num vetor que captura o sentido; uma regressão logística aprende as 77 intenções sobre
esses vetores. Não há ajuste do modelo grande, só da camada final: é o meio-termo entre o
clássico e o fine-tuning completo.
"""

from __future__ import annotations

import time

from sklearn.linear_model import LogisticRegression

from .dados import Exemplo

MODELO = "intfloat/multilingual-e5-small"


def rodar(treino: list[Exemplo], teste: list[Exemplo]) -> tuple[list[str], dict]:
    from sentence_transformers import SentenceTransformer

    codificador = SentenceTransformer(MODELO, device="cpu")
    # o e5 espera o prefixo "query: " nas frases a classificar
    inicio = time.perf_counter()
    x_treino = codificador.encode([f"query: {e.frase}" for e in treino], batch_size=64, normalize_embeddings=True)
    clf = LogisticRegression(C=20.0, max_iter=3000).fit(x_treino, [e.rotulo for e in treino])
    segundos_treino = time.perf_counter() - inicio

    inicio = time.perf_counter()
    x_teste = codificador.encode([f"query: {e.frase}" for e in teste], batch_size=64, normalize_embeddings=True)
    previsoes = [str(p) for p in clf.predict(x_teste)]
    ms = (time.perf_counter() - inicio) * 1000 / len(teste)
    return previsoes, {"parametros": f"{MODELO} (118 milhões, congelado) + regressão logística",
                       "segundos_treino": round(segundos_treino), "ms_por_exemplo": round(ms, 2),
                       "tamanho_mb": 470, "custo_por_1000": 0.0}
