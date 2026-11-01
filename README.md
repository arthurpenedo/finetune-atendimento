# finetune-atendimento

Classificação de intenção de atendimento bancário em português (77 intenções), comparando abordagens **com a mesma métrica**: ML clássico, embeddings, LLM sem treino, BERTimbau e Qwen com LoRA. A pergunta que o projeto responde: *fine-tuning vale o custo?*

Dados: [BANKING77 em português](https://huggingface.co/datasets/c2d-usp/banking77-pt-br) (CC BY 4.0), 10.003 frases de treino e 3.080 de teste.

## Resultado medido (linha de base)

| Abordagem | Acurácia | F1 macro | ms/exemplo | Tamanho | Custo/1.000 |
|---|---|---|---|---|---|
| TF-IDF + regressão logística (`resultados/classico.json`) | 89,84% | 89,88% | 0,16 | 25,2 MB | R$ 0 |

As outras abordagens (embeddings, zero-shot no Ollama, BERTimbau e Qwen+LoRA no Colab) têm código e comandos prontos, mas **ainda não têm resultado em `resultados/`**; só entram na tabela depois de medidas.

## Como rodar

```bash
pip install -e ".[dev]"
finetune-atendimento classico                    # baixa os dados na 1ª vez, treina e salva resultados/classico.json
finetune-atendimento embeddings                  # precisa do extra: pip install -e ".[embeddings]"
finetune-atendimento zeroshot --modelo qwen2.5:3b --por-classe 5   # precisa do Ollama rodando
finetune-atendimento importar-colab arquivo.txt  # importa a linha INICIO...FIM do notebook do Colab
finetune-atendimento relatorio --html site/index.html
python -m pytest -q                              # 8 testes
```

O fine-tuning (BERTimbau e Qwen com LoRA) roda em `notebooks/finetune_colab.ipynb` na GPU gratuita do Colab.

## Estrutura
- `src/finetune_atendimento/`: `dados.py` (download e leitura), `classico.py`, `embeddings.py`, `zeroshot.py`, `metricas.py` (acurácia, F1 macro, por classe), `relatorio.py`, `cli.py`.
- `resultados/`: um JSON por abordagem. `dados/`: CSVs baixados.
