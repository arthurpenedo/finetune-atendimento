"""Relatório HTML: todas as abordagens lado a lado, no teste completo e na mesma amostra."""

from __future__ import annotations

import html
from datetime import date

from .dados import Exemplo
from .metricas import avaliar

_ORDEM = ["ML clássico", "Embeddings", "LLM sem treino", "BERTimbau", "Qwen 2.5 0.5B"]

_CSS = """
:root{--fundo:#f7f7f8;--cartao:#fff;--texto:#1d1d22;--suave:#5c5c66;--borda:#e3e3e8;--destaque:#4c5fd5;--bom:#2f9e6e}
@media (prefers-color-scheme:dark){:root{--fundo:#141418;--cartao:#1d1d23;--texto:#ececf1;--suave:#a0a0ab;--borda:#2e2e37}}
*{box-sizing:border-box}body{margin:0;background:var(--fundo);color:var(--texto);font:15px/1.55 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:1060px;margin:0 auto;padding:32px 16px 64px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 12px}
.sub{color:var(--suave);margin:0 0 20px}table{width:100%;border-collapse:collapse;background:var(--cartao);
border:1px solid var(--borda);border-radius:12px;overflow:hidden}th,td{padding:9px 12px;border-bottom:1px solid var(--borda);
text-align:left;vertical-align:top}th{font-size:13px;color:var(--suave)}td.n{font-variant-numeric:tabular-nums;white-space:nowrap}
.barra{display:inline-block;height:9px;border-radius:5px;background:var(--destaque);margin-right:8px;vertical-align:middle}
.melhor{color:var(--bom);font-weight:700}.det{font-size:13px;color:var(--suave)}
footer{margin-top:48px;color:var(--suave);font-size:13px}a{color:var(--destaque)}
"""


def _ordem(e: dict) -> int:
    return next((i for i, p in enumerate(_ORDEM) if e["abordagem"].startswith(p)), 99)


def _ms(e: dict) -> str:
    texto = f"{e['ms_por_exemplo']:.2f} ms" if e["ms_por_exemplo"] < 10 else f"{e['ms_por_exemplo']:.0f} ms"
    if e.get("ms_por_exemplo_cpu"):
        texto += f" <span class='det'>({e['ms_por_exemplo_cpu']:.0f} ms na CPU)</span>"
    return texto


def gerar(execucoes: list[dict], teste: list[Exemplo]) -> str:
    execucoes = sorted(execucoes, key=_ordem)
    completas = [e for e in execucoes if "indices_amostra" not in e]
    amostra = next((e["indices_amostra"] for e in execucoes if "indices_amostra" in e), None)
    melhor_f1 = max(e["f1_macro"] for e in completas) if completas else 0

    linhas = ""
    for e in completas:
        destaque = " class='n melhor'" if e["f1_macro"] == melhor_f1 else " class='n'"
        tamanho = f"{e.get('tamanho_mb', 0):,.0f}".replace(",", ".")
        linhas += (f"<tr><td><b>{html.escape(e['abordagem'])}</b><div class='det'>{html.escape(e.get('parametros', ''))} · "
                   f"{html.escape(e.get('onde', ''))}</div></td><td{destaque}><span class='barra' style='width:"
                   f"{round(e['acuracia'] * 120)}px'></span>{e['acuracia']:.1%}</td><td{destaque}>{e['f1_macro']:.3f}</td>"
                   f"<td class='n'>{_ms(e)}</td><td class='n'>{e.get('segundos_treino', 0):.0f} s</td>"
                   f"<td class='n'>{tamanho} MB</td></tr>")

    bloco_amostra = ""
    if amostra:
        verdade = [teste[i].rotulo for i in amostra]
        linhas_a = ""
        for e in execucoes:
            prev = e["previsoes"] if "indices_amostra" in e else [e["previsoes"][i] for i in amostra]
            m = avaliar(verdade, prev)
            fora = e.get("respostas_fora_da_lista")
            extra = f"<div class='det'>{fora} respostas fora da lista de intenções</div>" if fora else ""
            linhas_a += (f"<tr><td>{html.escape(e['abordagem'])}{extra}</td><td class='n'>{m['acuracia']:.1%}</td>"
                         f"<td class='n'>{m['f1_macro']:.3f}</td><td class='n'>{_ms(e)}</td></tr>")
        bloco_amostra = (f"<h2>Todas as abordagens na mesma amostra ({len(amostra)} frases, 5 por intenção)</h2>"
                         "<p class='sub'>O LLM sem treino roda na CPU e leva segundos por frase, então foi medido numa amostra "
                         "estratificada. Para comparar de forma justa, as outras abordagens aparecem aqui medidas nas "
                         "mesmas frases.</p><table><tr><th>Abordagem</th><th>Acurácia</th><th>F1 macro</th><th>Tempo por frase</th>"
                         f"</tr>{linhas_a}</table>")

    melhor = max(completas, key=lambda e: e["f1_macro"]) if completas else None
    confusoes = ""
    if melhor:
        confusoes = "".join(f"<tr><td>{html.escape(c['verdade'])}</td><td>{html.escape(str(c['previsto']))}</td>"
                            f"<td class='n'>{c['vezes']}</td></tr>" for c in melhor["confusoes"][:10])
        confusoes = (f"<h2>Onde até o melhor modelo erra</h2><p class='sub'>{html.escape(melhor['abordagem'])}: os pares de "
                     "intenção mais confundidos. Quase sempre são intenções que um humano também confundiria.</p>"
                     f"<table><tr><th>Intenção real</th><th>Previsto</th><th>Vezes</th></tr>{confusoes}</table>")

    return f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>finetune-atendimento · resultados</title>
<style>{_CSS}</style></head><body><main>
<h1>finetune-atendimento · modelo pequeno treinado × LLM grande</h1>
<p class="sub">Classificação de 77 intenções de clientes de banco (BANKING77 em português, c2d-usp/banking77-pt-br, CC BY 4.0).
Teste: 3.080 frases nunca vistas no treino · {date.today():%d/%m/%Y}.</p>
<h2>No teste completo (3.080 frases)</h2>
<table><tr><th>Abordagem</th><th>Acurácia</th><th>F1 macro</th><th>Tempo por frase</th><th>Treino</th><th>Tamanho</th></tr>{linhas}</table>
<p class="det">F1 macro: média do F1 das 77 intenções, sem peso pelo tamanho; pune quem erra as intenções raras.</p>
{bloco_amostra}{confusoes}
<footer>Gerado pelo <a href="https://github.com/arthurpenedo/finetune-atendimento">finetune-atendimento</a>.</footer>
</main></body></html>"""
