"""BANKING77 em português (c2d-usp/banking77-pt-br, CC BY 4.0): 13.083 frases, 77 intenções."""

from __future__ import annotations

import csv
import urllib.request
from dataclasses import dataclass
from pathlib import Path

URL = "https://huggingface.co/datasets/c2d-usp/banking77-pt-br/resolve/main/{parte}.csv"
PASTA = Path("dados")


@dataclass(frozen=True)
class Exemplo:
    frase: str
    rotulo: str


def baixar(pasta: Path = PASTA) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    for parte in ("train", "test"):
        destino = pasta / f"{parte}.csv"
        if not destino.exists():
            urllib.request.urlretrieve(URL.format(parte=parte), destino)


def carregar(parte: str, pasta: Path = PASTA) -> list[Exemplo]:
    """parte: 'train' ou 'test'. Baixa na primeira vez."""
    baixar(pasta)
    with open(pasta / f"{parte}.csv", encoding="utf-8", newline="") as f:
        linhas = csv.DictReader(f)
        # o test.csv publicado termina com 2 linhas em branco; sem elas, são as 3.080 frases do BANKING77 original
        return [Exemplo(l["frase"].strip(), l["rotulo"].strip()) for l in linhas
                if (l.get("frase") or "").strip() and (l.get("rotulo") or "").strip()]


def rotulos(exemplos: list[Exemplo]) -> list[str]:
    return sorted({e.rotulo for e in exemplos})
