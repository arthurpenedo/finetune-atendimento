"""Linha de comando: rodar cada abordagem, importar o resultado do Colab e gerar o relatório."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import sys
from pathlib import Path

from . import dados, metricas

ARQUIVOS = {"bertimbau": "bertimbau.json", "qwen_lora": "qwen_lora.json"}


def _teste_e_rotulos():
    treino, teste = dados.carregar("train"), dados.carregar("test")
    return treino, teste, dados.rotulos(treino)


def _classico(args: argparse.Namespace) -> int:
    from . import classico

    treino, teste, _ = _teste_e_rotulos()
    prev, extras = classico.rodar(treino, teste)
    extras["onde"] = args.onde
    m = metricas.avaliar([e.rotulo for e in teste], prev)
    metricas.salvar(Path(args.pasta) / "classico.json", "ML clássico (TF-IDF + regressão logística)", m, extras, prev)
    print(f"clássico: acurácia {m['acuracia']:.4f} · F1 macro {m['f1_macro']:.4f}")
    return 0


def _embeddings(args: argparse.Namespace) -> int:
    from . import embeddings

    treino, teste, _ = _teste_e_rotulos()
    prev, extras = embeddings.rodar(treino, teste)
    extras["onde"] = args.onde
    m = metricas.avaliar([e.rotulo for e in teste], prev)
    metricas.salvar(Path(args.pasta) / "embeddings.json", "Embeddings multilíngues + regressão logística", m, extras, prev)
    print(f"embeddings: acurácia {m['acuracia']:.4f} · F1 macro {m['f1_macro']:.4f}")
    return 0


def _zeroshot(args: argparse.Namespace) -> int:
    from . import zeroshot

    _, teste, rotulos = _teste_e_rotulos()
    indices = zeroshot.amostra_estratificada(teste, args.por_classe)
    prev, extras = zeroshot.rodar(teste, rotulos, indices, args.modelo, args.url,
                                  lambda n, t: print(f"[{n}/{t}]", flush=True) if n % 25 == 0 else None)
    extras.update({"onde": args.onde, "custo_por_1000": 0.0, "indices_amostra": indices})
    m = metricas.avaliar([teste[i].rotulo for i in indices], prev)
    nome = args.modelo.replace(":", "-")
    metricas.salvar(Path(args.pasta) / f"zeroshot-{nome}.json", f"LLM sem treino ({args.modelo}, zero-shot)", m, extras, prev)
    print(f"zero-shot {args.modelo}: acurácia {m['acuracia']:.4f} · F1 macro {m['f1_macro']:.4f} "
          f"· {extras['respostas_fora_da_lista']} fora da lista")
    return 0


def _importar_colab(args: argparse.Namespace) -> int:
    """Lê a linha INICIO...FIM impressa pelo notebook e grava resultados/<abordagem>.json."""
    texto = Path(args.arquivo).read_text(encoding="utf-8") if args.arquivo else sys.stdin.read()
    carga = texto.split("INICIO", 1)[1].split("FIM", 1)[0].strip()
    resultados = json.loads(gzip.decompress(base64.b64decode(carga)))
    _, teste, _ = _teste_e_rotulos()
    verdade = [e.rotulo for e in teste]
    for chave, r in resultados.items():
        completo = {**r, **metricas.avaliar(verdade, r["previsoes"])}  # recalcula aqui, com o mesmo código
        destino = Path(args.pasta) / ARQUIVOS.get(chave, f"{chave}.json")
        destino.write_text(json.dumps(completo, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{chave}: acurácia {completo['acuracia']:.4f} · F1 macro {completo['f1_macro']:.4f} -> {destino}")
    return 0


def _relatorio(args: argparse.Namespace) -> int:
    from . import relatorio

    _, teste, _ = _teste_e_rotulos()
    execucoes = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.pasta).glob("*.json"))]
    Path(args.html).parent.mkdir(parents=True, exist_ok=True)
    Path(args.html).write_text(relatorio.gerar(execucoes, teste), encoding="utf-8")
    print(f"Relatório com {len(execucoes)} abordagens salvo em {args.html}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finetune-atendimento")
    parser.add_argument("--pasta", default="resultados")
    sub = parser.add_subparsers(dest="comando", required=True)
    for nome, func, onde in (("classico", _classico, "CPU"), ("embeddings", _embeddings, "CPU")):
        p = sub.add_parser(nome)
        p.add_argument("--onde", default=onde)
        p.set_defaults(func=func)
    p = sub.add_parser("zeroshot", help="LLM sem treino via Ollama, numa amostra estratificada")
    p.add_argument("--modelo", default="qwen2.5:3b")
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--por-classe", type=int, default=5)
    p.add_argument("--onde", default="CPU (Ollama)")
    p.set_defaults(func=_zeroshot)
    p = sub.add_parser("importar-colab", help="importa a linha INICIO...FIM do notebook")
    p.add_argument("arquivo", nargs="?")
    p.set_defaults(func=_importar_colab)
    p = sub.add_parser("relatorio")
    p.add_argument("--html", default="site/index.html")
    p.set_defaults(func=_relatorio)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
