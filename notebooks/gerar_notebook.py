"""Gera notebooks/finetune_colab.ipynb (rodar no Google Colab com GPU T4).

O notebook treina as abordagens que precisam de GPU e imprime, no fim, um JSON
compactado (gzip + base64) com métricas e previsões, no mesmo formato de
resultados/*.json, para ser copiado para o repositório.
"""

import json
from pathlib import Path

CELULAS = [
    ("markdown", """# finetune-atendimento · treino na GPU (Colab)

Treina e avalia, na **mesma tarefa e no mesmo teste** das outras abordagens do repositório:

- **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) ajustado como classificador de 77 intenções;
- **Qwen 2.5 0.5B Instruct** ajustado com **LoRA** para responder o nome da intenção.

Dados: [c2d-usp/banking77-pt-br](https://huggingface.co/datasets/c2d-usp/banking77-pt-br) (CC BY 4.0).

**Como rodar:** Ambiente de execução → Alterar o tipo de ambiente → **GPU T4** → Executar tudo.
Leva ~25 minutos. A última célula imprime o resultado para copiar."""),
    ("code", """!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install -U transformers==4.46.3 peft==0.13.2 accelerate==1.1.1 datasets==3.1.0 scikit-learn"""),
    ("code", """import csv, json, time, random, gzip, base64, urllib.request, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
URL = "https://huggingface.co/datasets/c2d-usp/banking77-pt-br/resolve/main/{}.csv"
def carregar(parte):
    urllib.request.urlretrieve(URL.format(parte), f"{parte}.csv")
    with open(f"{parte}.csv", encoding="utf-8", newline="") as f:
        return [(l["frase"].strip(), l["rotulo"].strip()) for l in csv.DictReader(f)
                if (l.get("frase") or "").strip() and (l.get("rotulo") or "").strip()]
treino, teste = carregar("train"), carregar("test")
ROTULOS = sorted({r for _, r in treino}); ID = {r: i for i, r in enumerate(ROTULOS)}
print(len(treino), len(teste), len(ROTULOS), "intenções")
GPU = torch.cuda.get_device_name(0)

def metricas(verdade, previsto):
    from sklearn.metrics import accuracy_score, f1_score
    rot = [p if p in ID else "__nenhum__" for p in previsto]
    return {"acuracia": round(accuracy_score(verdade, rot), 4),
            "f1_macro": round(f1_score(verdade, rot, labels=ROTULOS, average="macro", zero_division=0), 4)}
RESULTADOS = {}"""),
    ("markdown", "## 1 · BERTimbau ajustado como classificador"),
    ("code", """from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
NOME = "neuralmind/bert-base-portuguese-cased"
tok = AutoTokenizer.from_pretrained(NOME)
def ds(pares):
    d = Dataset.from_dict({"text": [f for f, _ in pares], "label": [ID[r] for _, r in pares]})
    return d.map(lambda b: tok(b["text"], truncation=True, max_length=64), batched=True)
dtreino, dteste = ds(treino), ds(teste)
modelo = AutoModelForSequenceClassification.from_pretrained(NOME, num_labels=len(ROTULOS))
args = TrainingArguments("bert", num_train_epochs=4, per_device_train_batch_size=32, per_device_eval_batch_size=128,
                         learning_rate=5e-5, warmup_ratio=0.1, weight_decay=0.01, fp16=True, logging_steps=100,
                         save_strategy="no", report_to=[], seed=42)
inicio = time.time()
Trainer(model=modelo, args=args, train_dataset=dtreino, tokenizer=tok).train()
seg_treino = time.time() - inicio
modelo.eval()
prev, inicio = [], time.time()
with torch.no_grad():
    for i in range(0, len(teste), 128):
        lote = tok([f for f, _ in teste[i:i+128]], padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
        prev += [ROTULOS[j] for j in modelo(**lote).logits.argmax(-1).tolist()]
ms = (time.time() - inicio) * 1000 / len(teste)
# latência em CPU, uma frase por vez (o cenário de um servidor simples sem GPU)
modelo_cpu = modelo.float().to("cpu"); t0 = time.time()
with torch.no_grad():
    for f, _ in teste[:100]: modelo_cpu(**tok(f, return_tensors="pt"))
ms_cpu = (time.time() - t0) * 1000 / 100
m = metricas([r for _, r in teste], prev)
RESULTADOS["bertimbau"] = {"abordagem": "BERTimbau ajustado (fine-tuning)", **m, "onde": f"Colab {GPU}",
    "parametros": "110 milhões", "segundos_treino": round(seg_treino), "ms_por_exemplo": round(ms, 3),
    "ms_por_exemplo_cpu": round(ms_cpu, 1), "tamanho_mb": 420, "custo_por_1000": 0.0, "previsoes": prev}
print(m, f"treino {seg_treino:.0f}s · {ms:.2f} ms/frase na GPU · {ms_cpu:.0f} ms/frase na CPU")
del modelo, modelo_cpu; torch.cuda.empty_cache()"""),
    ("markdown", "## 2 · Qwen 2.5 0.5B ajustado com LoRA (o \"LLM pequeno especializado\")"),
    ("code", """from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
NOME = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(NOME); tok.padding_side = "left"
llm = AutoModelForCausalLM.from_pretrained(NOME, torch_dtype=torch.float32).to("cuda")  # T4 não tem bf16: pesos em fp32, treino em fp16 misto
llm = get_peft_model(llm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
llm.print_trainable_parameters()
INSTRUCAO = "Classifique a intenção da mensagem de um cliente de banco. Responda só com o nome da intenção."
def prompt(frase):
    return tok.apply_chat_template([{"role": "system", "content": INSTRUCAO}, {"role": "user", "content": frase}],
                                   tokenize=False, add_generation_prompt=True)
def exemplo(frase, rotulo):
    p, alvo = tok(prompt(frase))["input_ids"], tok(rotulo + tok.eos_token)["input_ids"]
    return {"input_ids": p + alvo, "labels": [-100] * len(p) + alvo}  # só a resposta entra na perda
dtreino = Dataset.from_list([exemplo(f, r) for f, r in treino])
def juntar(lote):
    n = max(len(x["input_ids"]) for x in lote)
    ids = [[tok.pad_token_id] * (n - len(x["input_ids"])) + x["input_ids"] for x in lote]
    lab = [[-100] * (n - len(x["labels"])) + x["labels"] for x in lote]
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
            "attention_mask": torch.tensor([[0] * (n - len(x["input_ids"])) + [1] * len(x["input_ids"]) for x in lote])}
args = TrainingArguments("qwen", num_train_epochs=2, per_device_train_batch_size=16, learning_rate=2e-4,
                         warmup_ratio=0.05, lr_scheduler_type="cosine", fp16=True, logging_steps=100,
                         save_strategy="no", report_to=[], seed=42, remove_unused_columns=False)
inicio = time.time()
Trainer(model=llm, args=args, train_dataset=dtreino, data_collator=juntar).train()
seg_treino = time.time() - inicio
llm.eval(); prev, inicio = [], time.time()
with torch.no_grad():
    for i in range(0, len(teste), 64):
        lote = tok([prompt(f) for f, _ in teste[i:i+64]], return_tensors="pt", padding=True).to("cuda")
        saida = llm.generate(**lote, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
        prev += [t.strip() for t in tok.batch_decode(saida[:, lote["input_ids"].shape[1]:], skip_special_tokens=True)]
ms = (time.time() - inicio) * 1000 / len(teste)
fora = sum(p not in ID for p in prev)
m = metricas([r for _, r in teste], prev)
RESULTADOS["qwen_lora"] = {"abordagem": "Qwen 2.5 0.5B ajustado com LoRA", **m, "onde": f"Colab {GPU}",
    "parametros": "494 milhões (8,8 milhões treinados com LoRA)", "segundos_treino": round(seg_treino),
    "ms_por_exemplo": round(ms, 3), "tamanho_mb": 990, "custo_por_1000": 0.0, "respostas_fora_da_lista": fora,
    "previsoes": [p if p in ID else None for p in prev]}
print(m, f"treino {seg_treino:.0f}s · {ms:.1f} ms/frase · {fora} respostas fora da lista de intenções")"""),
    ("markdown", "## 3 · Resultado para copiar\n\nCopie a linha entre `INICIO` e `FIM` e cole na conversa."),
    ("code", """bruto = json.dumps(RESULTADOS, ensure_ascii=False).encode()
print("INICIO" + base64.b64encode(gzip.compress(bruto)).decode() + "FIM")
print({k: (v["acuracia"], v["f1_macro"]) for k, v in RESULTADOS.items()})"""),
]


def main() -> None:
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "cells": []}
    for tipo, fonte in CELULAS:
        celula = {"cell_type": tipo, "metadata": {}, "source": fonte}
        if tipo == "code":
            celula.update({"execution_count": None, "outputs": []})
        nb["cells"].append(celula)
    destino = Path(__file__).with_name("finetune_colab.ipynb")
    destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"salvo em {destino}")


if __name__ == "__main__":
    main()
