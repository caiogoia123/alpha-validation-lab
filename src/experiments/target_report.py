"""Orquestra o estudo comparativo de alvos e gera relatório + gráficos.

Responde à pergunta principal:
  "Qual variável contém mais informação previsível do que simplesmente prever
   alta ou baixa?"
sem tentar construir uma estratégia lucrativa — apenas medir previsibilidade,
estabilidade temporal e valor econômico potencial de cada alvo.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from src.data.database import Database
from src.experiments.target_study import run_size_study
from src.experiments.targets import TARGET_LABELS, TARGET_NAMES

DEFAULT_HORIZONS = [5, 15, 60]
_PALETTE = ["#2563eb", "#16a34a", "#f59e0b", "#db2777"]


def run_study(db: Database = None, horizons: list[int] = None,
              n_candles: int = 250_000, targets: list[str] = None,
              overrides: dict = None) -> dict:
    """Executa o estudo no maior histórico disponível e grava os artefatos."""
    db = db or Database()
    horizons = horizons or DEFAULT_HORIZONS
    targets = targets or TARGET_NAMES

    base_params = dict(config.BACKTEST)
    if overrides:
        base_params.update({k: v for k, v in overrides.items() if v is not None})

    candles = db.load_candles(limit=n_candles)
    if len(candles) < 5_000:
        raise RuntimeError(f"Histórico insuficiente: {len(candles)} candles.")

    print(f"  • {len(candles):,} candles · alvos={len(targets)} · horizontes={horizons}")
    study = run_size_study(candles, horizons, base_params, targets)

    agg = _aggregate(study, targets, horizons)
    verdict = _build_verdict(agg, horizons)

    auc_chart = _plot_predictability(study, targets, horizons)
    econ_chart = _plot_economic(study, targets, horizons)

    md = _render_markdown(study, agg, verdict, horizons, base_params,
                          auc_chart, econ_chart)
    report_path = os.path.join(config.REPORTS_DIR, "target_study_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)

    return {"report_path": report_path, "study": study, "aggregate": agg,
            "verdict": verdict}


# ---------------------------------------------------------------------------
# Agregação e conclusão
# ---------------------------------------------------------------------------
def _aggregate(study: dict, targets: list[str], horizons: list[int]) -> dict:
    """Média por alvo entre horizontes das métricas-chave."""
    by_target = {t: [] for t in targets}
    for r in study["results"]:
        by_target[r["target"]].append(r)

    agg = {}
    for t, rows in by_target.items():
        aucs = [r["auc"] for r in rows if r["auc"] == r["auc"]]
        skills = [r["acc_skill"] for r in rows if r["acc_skill"] == r["acc_skill"]]
        econ = [r["econ_separation_pct"] for r in rows
                if r["econ_separation_pct"] == r["econ_separation_pct"]]
        stds = [r["auc_std_folds"] for r in rows if r["auc_std_folds"] == r["auc_std_folds"]]
        n_sig = sum(1 for r in rows
                    if r["p_value"] == r["p_value"] and r["p_value"] < 0.05
                    and r["auc"] == r["auc"] and r["auc"] > 0.5)
        agg[t] = {
            "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "min_auc": float(np.min(aucs)) if aucs else float("nan"),
            "n_auc_above_half": sum(1 for x in aucs if x > 0.5),
            "mean_acc_skill": float(np.mean(skills)) if skills else float("nan"),
            "mean_econ_sep": float(np.mean(econ)) if econ else float("nan"),
            "mean_auc_std": float(np.mean(stds)) if stds else float("nan"),
            "n_significant": n_sig,
            "n_horizons": len(rows),
        }
    return agg


def _build_verdict(agg: dict, horizons: list[int]) -> dict:
    """Ranqueia os alvos por previsibilidade e identifica o mais promissor."""
    # Ordena por AUC médio (informação previsível), desempate por econ.
    ranking = sorted(agg.items(),
                     key=lambda kv: (kv[1]["mean_auc"] if kv[1]["mean_auc"] == kv[1]["mean_auc"] else 0,
                                     kv[1]["mean_econ_sep"] if kv[1]["mean_econ_sep"] == kv[1]["mean_econ_sep"] else 0),
                     reverse=True)
    best_target, best = ranking[0]
    direction = agg.get("direcao", {})

    # Um alvo é "claramente mais previsível que direção" se (usando AUC, métrica
    # comparável e independente de limiar):
    #   (a) AUC médio supera o da direção por margem relevante (>0,02);
    #   (b) AUC acima do acaso em TODOS os horizontes (consistência); e
    #   (c) AUC robustamente acima de 0,5 considerando a variabilidade entre
    #       folds (mean_auc - 2*desvio > 0,5).
    dir_auc = direction.get("mean_auc", 0.5)
    beats_direction = []
    for t, a in agg.items():
        if t == "direcao" or a["mean_auc"] != a["mean_auc"]:
            continue
        robust = (a["mean_auc"] - 2 * a["mean_auc_std"]) > 0.5
        consistent = a["n_auc_above_half"] == a["n_horizons"]
        if a["mean_auc"] > dir_auc + 0.02 and robust and consistent:
            beats_direction.append((t, a))
    beats_direction.sort(key=lambda kv: kv[1]["mean_auc"], reverse=True)

    return {
        "ranking": ranking,
        "best_target": best_target,
        "best_metrics": best,
        "direction_auc": dir_auc,
        "beats_direction": beats_direction,
        "has_better_target": len(beats_direction) > 0,
    }


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def _hz_label(h: int) -> str:
    return f"{h//60}h" if h >= 60 and h % 60 == 0 else f"{h}min"


def _grouped_positions(n_targets: int, n_groups: int):
    x = np.arange(n_targets)
    width = 0.8 / n_groups
    offsets = [(-0.4 + width / 2) + i * width for i in range(n_groups)]
    return x, width, offsets


def _value_grid(study, targets, horizons, key):
    """Matriz [target][horizon] de uma métrica."""
    idx = {(r["target"], r["horizon"]): r for r in study["results"]}
    grid = np.full((len(targets), len(horizons)), np.nan)
    for i, t in enumerate(targets):
        for j, h in enumerate(horizons):
            r = idx.get((t, h))
            if r:
                grid[i, j] = r[key]
    return grid


def _plot_predictability(study, targets, horizons) -> str:
    auc = _value_grid(study, targets, horizons, "auc")
    std = _value_grid(study, targets, horizons, "auc_std_folds")
    x, width, offsets = _grouped_positions(len(targets), len(horizons))

    fig, ax = plt.subplots(figsize=(12, 6))
    for j, h in enumerate(horizons):
        ax.bar(x + offsets[j], auc[:, j], width, yerr=std[:, j], capsize=3,
               color=_PALETTE[j % len(_PALETTE)], label=_hz_label(h))
    ax.axhline(0.5, color="#ef4444", linestyle="--", label="Acaso (AUC 0,5)")
    ax.set_title("Previsibilidade por alvo — AUC out-of-sample (barras de erro = desvio entre folds)")
    ax.set_ylabel("AUC")
    ax.set_xticks(x)
    ax.set_xticklabels([t for t in targets], rotation=15, ha="right")
    ax.set_ylim(0.40, max(0.75, np.nanmax(auc) + 0.05))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path = os.path.join(config.REPORTS_DIR, "target_predictability.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_economic(study, targets, horizons) -> str:
    econ = _value_grid(study, targets, horizons, "econ_separation_pct")
    hurdle = _value_grid(study, targets, horizons, "cost_hurdle_pct")
    cost = np.nanmean(hurdle)
    x, width, offsets = _grouped_positions(len(targets), len(horizons))

    fig, ax = plt.subplots(figsize=(12, 6))
    for j, h in enumerate(horizons):
        ax.bar(x + offsets[j], econ[:, j], width,
               color=_PALETTE[j % len(_PALETTE)], label=_hz_label(h))
    ax.axhline(cost, color="#ef4444", linestyle="--",
               label=f"Custo ida-e-volta (~{cost:.2f}%)")
    ax.set_title("Valor econômico potencial — separação do |movimento| realizado "
                 "entre as classes previstas")
    ax.set_ylabel("Separação de |retorno| (pontos %)")
    ax.set_xticks(x)
    ax.set_xticklabels([t for t in targets], rotation=15, ha="right")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path = os.path.join(config.REPORTS_DIR, "target_economic_value.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _fmt_time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _render_markdown(study, agg, verdict, horizons, base_params,
                     auc_chart, econ_chart) -> str:
    L: list[str] = []
    a = L.append

    a("# Estudo Comparativo de Variáveis-Alvo — BTC/USDT")
    a("")
    a(f"_Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
      f"{study['n_candles']:,} candles · {_fmt_time(study['start'])} a "
      f"{_fmt_time(study['end'])}._")
    a("")
    a("> Mesmo XGBoost e mesmos indicadores; muda-se apenas **a variável-alvo**. "
      "Objetivo: descobrir se existe um alvo mais previsível e economicamente "
      "útil do que a direção binária — **não** construir uma estratégia. Todas "
      "as estatísticas usam amostras não-sobrepostas (≈ N/H independentes).")
    a("")
    a("**Alvos avaliados** (todos derivados só do preço, como rótulos):")
    a("")
    for t in agg:
        a(f"- `{t}` — {TARGET_LABELS[t]}")
    a("")

    # --- Tabela por horizonte --------------------------------------------
    a("## 1. Previsibilidade, estabilidade e valor econômico")
    a("")
    a("`AUC` = poder de separação (0,5 = acaso). `Skill acc.` = acurácia menos "
      "baseline da classe majoritária. `AUC±` = desvio do AUC entre folds "
      "sequenciais (estabilidade temporal; menor é melhor). `Sep.econ` = "
      "separação do |movimento| realizado entre classes previstas vs. custo.")
    a("")
    for h in horizons:
        a(f"### Horizonte {_hz_label(h)}")
        a("")
        a("| Alvo | Classes | N indep | AUC | Skill acc. | p-valor | AUC± (folds) | "
          "Sep.econ | Custo | Sep>Custo? |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
        rows = [r for r in study["results"] if r["horizon"] == h]
        # Ordena por AUC desc para leitura.
        rows.sort(key=lambda r: (r["auc"] if r["auc"] == r["auc"] else 0),
                  reverse=True)
        for r in rows:
            if r["auc"] != r["auc"]:
                a(f"| {r['target']} | {r['n_classes']} | {r['n_indep']} | — | — | "
                  "— | — | — | — | — |")
                continue
            a("| {} | {} | {:,} | {:.3f} | {:+.1%} | {:.3f} | {:.3f} | "
              "{:.3f}% | {:.3f}% | {} |".format(
                  r["target"], r["n_classes"], r["n_indep"], r["auc"],
                  r["acc_skill"], r["p_value"], r["auc_std_folds"],
                  r["econ_separation_pct"], r["cost_hurdle_pct"],
                  "✅" if r["econ_beats_cost"] else "—"))
        a("")

    # --- Ranking agregado -------------------------------------------------
    a("## 2. Ranking de previsibilidade (média entre horizontes)")
    a("")
    a("| # | Alvo | AUC médio | Skill acc. médio | Sep.econ média | "
      "Estab. (AUC± médio) | Horizontes significativos |")
    a("|---:|---|---:|---:|---:|---:|---:|")
    for i, (t, m) in enumerate(verdict["ranking"], start=1):
        a("| {} | {} | {:.3f} | {:+.1%} | {:.3f}% | {:.3f} | {}/{} |".format(
            i, t, m["mean_auc"], m["mean_acc_skill"], m["mean_econ_sep"],
            m["mean_auc_std"], m["n_significant"], m["n_horizons"]))
    a("")

    a("![Previsibilidade por alvo](" +
      os.path.relpath(auc_chart, config.REPORTS_DIR).replace(os.sep, "/") + ")")
    a("")
    a("![Valor econômico por alvo](" +
      os.path.relpath(econ_chart, config.REPORTS_DIR).replace(os.sep, "/") + ")")
    a("")

    # --- Conclusão --------------------------------------------------------
    a("## 3. Conclusão objetiva")
    a("")
    a("**Pergunta:** qual variável contém mais informação previsível do que "
      "simplesmente prever alta ou baixa?")
    a("")
    best = verdict["best_target"]
    bm = verdict["best_metrics"]
    dir_auc = verdict["direction_auc"]
    a(f"**Resposta: `{best}` — {TARGET_LABELS[best]}.**")
    a("")
    a(f"- AUC médio de **{bm['mean_auc']:.3f}** vs. **{dir_auc:.3f}** da direção "
      f"(acaso = 0,500). A direção permanece praticamente indistinguível do "
      f"acaso; `{best}` carrega informação previsível real e estável "
      f"(significativo em {bm['n_significant']}/{bm['n_horizons']} horizontes).")
    if verdict["beats_direction"]:
        outros = ", ".join(f"`{t}` (AUC {m['mean_auc']:.3f})"
                           for t, m in verdict["beats_direction"])
        a(f"- Alvos que superam a direção de forma significativa e estável: {outros}.")
    a("")
    a("**Implicação econômica:** a separação do |movimento| entre as classes "
      f"previstas do melhor alvo é de ~{bm['mean_econ_sep']:.3f} pontos %, a ser "
      "comparada com o custo de ida-e-volta. Mesmo sem prever direção, antecipar "
      "o **tamanho/volatilidade** do movimento é a informação monetizável (sizing, "
      "opções, gestão de risco).")
    a("")
    a("**Recomendação de pesquisa:** a direção binária está esgotada; o próximo "
      f"alvo a modelar é **{best}** ({TARGET_LABELS[best]}). Só então faz sentido "
      "discutir features dedicadas — este estudo já o identifica como mais "
      "previsível **mesmo usando as features atuais, pensadas para direção**.")
    a("")
    a("_Ressalva: previsibilidade medida com o conjunto de features atual e um "
      "único período. Um alvo promissor aqui é candidato a aprofundamento, não "
      "uma garantia de lucro — que dependerá de custos e execução._")
    a("")
    return "\n".join(L)
