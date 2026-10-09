"""Métricas comuns a todas as abordagens e o formato do arquivo de resultado.

Todas as abordagens gravam o mesmo JSON (resultados/<nome>.json), venham do PC, do
GitHub Actions ou do Colab, para o relatório comparar tudo lado a lado.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def avaliar(verdade: list[str], previsto: list[str | None]) -> dict:
    """Acurácia, F1 macro (média simples entre as 77 classes, que pune errar as classes raras),
    F1 por classe e os pares de intenção mais confundidos."""
    if len(verdade) != len(previsto):
        raise ValueError("tamanhos diferentes")
    classes = sorted(set(verdade))
    vp, fp, fn = Counter(), Counter(), Counter()
    confusoes: Counter = Counter()
    for v, p in zip(verdade, previsto):
        if v == p:
            vp[v] += 1
        else:
            fn[v] += 1
            if p is not None:
                fp[p] += 1
            confusoes[(v, p or "(sem resposta)")] += 1
    f1 = {}
    for c in classes:
        prec = vp[c] / (vp[c] + fp[c]) if vp[c] + fp[c] else 0.0
        rec = vp[c] / (vp[c] + fn[c]) if vp[c] + fn[c] else 0.0
        f1[c] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "exemplos": len(verdade),
        "acuracia": round(sum(vp.values()) / len(verdade), 4),
        "f1_macro": round(sum(f1.values()) / len(classes), 4),
        "sem_resposta": sum(1 for p in previsto if p is None),
        "f1_por_classe": {c: round(v, 4) for c, v in f1.items()},
        "confusoes": [{"verdade": v, "previsto": p, "vezes": n} for (v, p), n in confusoes.most_common(15)],
    }


def salvar(caminho: str | Path, abordagem: str, metricas: dict, extras: dict, previsoes: list[str | None]) -> dict:
    dados = {"abordagem": abordagem, **metricas, **extras, "previsoes": previsoes}
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    Path(caminho).write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    return dados
