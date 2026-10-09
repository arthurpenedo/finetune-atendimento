"""Abordagem 3: LLM sem treino (zero-shot). O modelo recebe a lista das 77 intenções e escolhe uma.

É o que muita empresa faz primeiro: "é só perguntar para o LLM". Funciona sem nenhum dado
rotulado, mas cada frase é uma chamada de modelo grande, com a lista inteira no prompt.
Respostas fora da lista são aproximadas pelo nome mais parecido (difflib) e contadas.
"""

from __future__ import annotations

import difflib
import json
import random
import time
import urllib.request
from collections import defaultdict

from .dados import Exemplo

PROMPT = """Você classifica mensagens de clientes de um banco digital em UMA intenção.
Responda APENAS com o nome exato de uma intenção da lista, sem explicação.

Intenções:
{lista}"""


def amostra_estratificada(teste: list[Exemplo], por_classe: int = 5, semente: int = 42) -> list[int]:
    """Índices de `por_classe` exemplos de cada intenção (sempre os mesmos, pela semente)."""
    por_rotulo: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(teste):
        por_rotulo[e.rotulo].append(i)
    rnd = random.Random(semente)
    indices = []
    for rotulo in sorted(por_rotulo):
        indices += sorted(rnd.sample(por_rotulo[rotulo], min(por_classe, len(por_rotulo[rotulo]))))
    return indices


def normalizar_resposta(resposta: str, rotulos: list[str]) -> tuple[str | None, bool]:
    """(rótulo, estava_na_lista). Tenta o nome exato; senão, o mais parecido acima de 0,75."""
    limpa = resposta.strip().strip("`'\".").split("\n")[0].strip().lower().replace(" ", "_")
    if limpa in rotulos:
        return limpa, True
    parecidos = difflib.get_close_matches(limpa, rotulos, n=1, cutoff=0.75)
    return (parecidos[0] if parecidos else None), False


def classificar(frase: str, rotulos: list[str], modelo: str, url: str) -> str:
    corpo = {"model": modelo, "stream": False, "options": {"temperature": 0, "seed": 42, "num_predict": 24},
             "messages": [{"role": "system", "content": PROMPT.format(lista="\n".join(rotulos))},
                          {"role": "user", "content": frase}]}
    req = urllib.request.Request(f"{url}/api/chat", json.dumps(corpo).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"]


def rodar(teste: list[Exemplo], rotulos: list[str], indices: list[int], modelo: str = "qwen2.5:3b",
          url: str = "http://localhost:11434", progresso=None) -> tuple[list[str | None], dict]:
    previsoes, fora = [], 0
    inicio = time.perf_counter()
    for n, i in enumerate(indices, 1):
        rotulo, na_lista = normalizar_resposta(classificar(teste[i].frase, rotulos, modelo, url), rotulos)
        fora += not na_lista
        previsoes.append(rotulo)
        if progresso:
            progresso(n, len(indices))
    ms = (time.perf_counter() - inicio) * 1000 / len(indices)
    return previsoes, {"parametros": f"{modelo} (3 bilhões), sem treino", "ms_por_exemplo": round(ms, 1),
                       "respostas_fora_da_lista": fora, "tamanho_mb": 1900, "segundos_treino": 0}
