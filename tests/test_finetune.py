import base64
import gzip
import json

import pytest

from finetune_atendimento import dados, metricas, relatorio, zeroshot
from finetune_atendimento.cli import main
from finetune_atendimento.dados import Exemplo

TESTE = [Exemplo(f"frase {i}", r) for i, r in enumerate(["a", "a", "b", "b", "c", "c"])]


def test_metricas_acuracia_f1_e_confusoes():
    m = metricas.avaliar(["a", "a", "b", "c"], ["a", "b", "b", None])
    assert m["acuracia"] == 0.5 and m["sem_resposta"] == 1
    assert m["f1_por_classe"]["c"] == 0.0
    assert {"verdade": "a", "previsto": "b", "vezes": 1} in m["confusoes"]


def test_metricas_tamanhos_diferentes_levanta_erro():
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        metricas.avaliar(["a", "b"], ["a"])


def test_metricas_salvar_grava_json_legivel(tmp_path):
    caminho = tmp_path / "novo" / "x.json"  # a pasta ainda não existe
    m = metricas.avaliar(["a", "b"], ["a", "b"])
    dados = metricas.salvar(caminho, "teste", m, {"ms_por_exemplo": 2}, ["a", "b"])
    lido = json.loads(caminho.read_text(encoding="utf-8"))
    assert lido == dados
    assert lido["abordagem"] == "teste" and lido["acuracia"] == 1.0 and lido["previsoes"] == ["a", "b"]


def test_carregar_ignora_linhas_em_branco_e_tira_espacos(tmp_path):
    (tmp_path / "train.csv").write_text("frase,rotulo\n oi ,a\n,\nsem rotulo,\ntchau,b\n", encoding="utf-8")
    (tmp_path / "test.csv").write_text("frase,rotulo\nx,c\n", encoding="utf-8")
    treino = dados.carregar("train", tmp_path)  # arquivos já existem: não baixa nada
    assert treino == [Exemplo("oi", "a"), Exemplo("tchau", "b")]
    assert dados.rotulos(treino + dados.carregar("test", tmp_path)) == ["a", "b", "c"]


def test_amostra_estratificada_e_deterministica():
    a = zeroshot.amostra_estratificada(TESTE, por_classe=1)
    assert a == zeroshot.amostra_estratificada(TESTE, por_classe=1)
    assert sorted(TESTE[i].rotulo for i in a) == ["a", "b", "c"]


def test_normalizar_resposta_do_llm():
    rotulos = ["chegada_do_cartão", "cancelar_transferência"]
    assert zeroshot.normalizar_resposta("chegada_do_cartão", rotulos) == ("chegada_do_cartão", True)
    assert zeroshot.normalizar_resposta("`Cancelar transferência`.", rotulos) == ("cancelar_transferência", True)
    assert zeroshot.normalizar_resposta("chegada do cartao", rotulos) == ("chegada_do_cartão", False)
    assert zeroshot.normalizar_resposta("não sei", rotulos) == (None, False)


def test_relatorio_compara_na_mesma_amostra():
    completa = {"abordagem": "ML clássico (x)", **metricas.avaliar([e.rotulo for e in TESTE], ["a"] * 6),
                "previsoes": ["a"] * 6, "ms_por_exemplo": 0.1, "segundos_treino": 1, "tamanho_mb": 1200}
    amostra = [0, 2, 4]
    llm = {"abordagem": "LLM sem treino (y)", **metricas.avaliar(["a", "b", "c"], ["a", "b", "c"]),
           "previsoes": ["a", "b", "c"], "indices_amostra": amostra, "ms_por_exemplo": 900, "respostas_fora_da_lista": 2}
    pagina = relatorio.gerar([llm, completa], TESTE)
    assert "mesma amostra (3 frases" in pagina and "1.200 MB" in pagina and "2 respostas fora" in pagina


def test_importar_resultado_do_colab(tmp_path, monkeypatch):
    from finetune_atendimento import cli

    monkeypatch.setattr(cli, "_teste_e_rotulos", lambda: (TESTE, TESTE, ["a", "b", "c"]))
    carga = {"bertimbau": {"abordagem": "BERTimbau ajustado", "previsoes": ["a", "a", "b", "b", "c", "a"],
                           "ms_por_exemplo": 0.4, "acuracia": 0}}
    linha = "INICIO" + base64.b64encode(gzip.compress(json.dumps(carga).encode())).decode() + "FIM"
    (tmp_path / "saida.txt").write_text(f"lixo antes\n{linha}\nlixo depois", encoding="utf-8")
    assert main(["--pasta", str(tmp_path), "importar-colab", str(tmp_path / "saida.txt")]) == 0
    salvo = json.loads((tmp_path / "bertimbau.json").read_text(encoding="utf-8"))
    assert round(salvo["acuracia"], 4) == round(5 / 6, 4)  # recalculado aqui, não confiando no número do Colab
