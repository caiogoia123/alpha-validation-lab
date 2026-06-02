# Estudo Comparativo de Variáveis-Alvo — BTC/USDT

_Gerado em 2026-06-01 20:53:54 · 250,000 candles · 2025-12-10 08:48 UTC a 2026-06-01 23:27 UTC._

> Mesmo XGBoost e mesmos indicadores; muda-se apenas **a variável-alvo**. Objetivo: descobrir se existe um alvo mais previsível e economicamente útil do que a direção binária — **não** construir uma estratégia. Todas as estatísticas usam amostras não-sobrepostas (≈ N/H independentes).

**Alvos avaliados** (todos derivados só do preço, como rótulos):

- `direcao` — Direção (alta/baixa)
- `magnitude` — Magnitude do retorno (terços)
- `volatilidade` — Volatilidade futura (alta/baixa)
- `mov_absoluto` — Movimento absoluto |ret| (grande/pequeno)
- `regime` — Regime de mercado (5 classes)

## 1. Previsibilidade, estabilidade e valor econômico

`AUC` = poder de separação (0,5 = acaso). `Skill acc.` = acurácia menos baseline da classe majoritária. `AUC±` = desvio do AUC entre folds sequenciais (estabilidade temporal; menor é melhor). `Sep.econ` = separação do |movimento| realizado entre classes previstas vs. custo.

### Horizonte 5min

| Alvo | Classes | N indep | AUC | Skill acc. | p-valor | AUC± (folds) | Sep.econ | Custo | Sep>Custo? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| volatilidade | 2 | 14,999 | 0.770 | +17.5% | 0.000 | 0.017 | 0.038% | 0.120% | — |
| mov_absoluto | 2 | 14,999 | 0.653 | +5.9% | 0.000 | 0.019 | 0.032% | 0.120% | — |
| regime | 5 | 14,998 | 0.648 | +15.2% | 0.000 | 0.021 | 0.038% | 0.120% | — |
| magnitude | 3 | 14,999 | 0.595 | +5.2% | 0.000 | 0.012 | 0.043% | 0.120% | — |
| direcao | 2 | 14,999 | 0.521 | +1.0% | 0.011 | 0.011 | 0.012% | 0.120% | — |

### Horizonte 15min

| Alvo | Classes | N indep | AUC | Skill acc. | p-valor | AUC± (folds) | Sep.econ | Custo | Sep>Custo? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| volatilidade | 2 | 4,999 | 0.786 | +17.0% | 0.000 | 0.023 | 0.054% | 0.120% | — |
| regime | 5 | 4,999 | 0.660 | +15.3% | 0.000 | 0.014 | 0.054% | 0.120% | — |
| mov_absoluto | 2 | 4,999 | 0.646 | +4.2% | 0.000 | 0.022 | 0.049% | 0.120% | — |
| magnitude | 3 | 4,999 | 0.579 | +4.1% | 0.000 | 0.016 | 0.067% | 0.120% | — |
| direcao | 2 | 4,999 | 0.513 | +0.6% | 0.379 | 0.022 | 0.018% | 0.120% | — |

### Horizonte 1h

| Alvo | Classes | N indep | AUC | Skill acc. | p-valor | AUC± (folds) | Sep.econ | Custo | Sep>Custo? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| volatilidade | 2 | 1,249 | 0.763 | +2.0% | 0.149 | 0.027 | 0.091% | 0.120% | — |
| mov_absoluto | 2 | 1,249 | 0.621 | +1.2% | 0.412 | 0.035 | 0.092% | 0.120% | — |
| regime | 5 | 1,249 | 0.595 | +2.1% | 0.109 | 0.033 | 0.217% | 0.120% | ✅ |
| magnitude | 3 | 1,249 | 0.541 | -1.1% | 0.423 | 0.020 | 0.096% | 0.120% | — |
| direcao | 2 | 1,249 | 0.463 | -0.8% | 0.576 | 0.020 | 0.002% | 0.120% | — |

## 2. Ranking de previsibilidade (média entre horizontes)

| # | Alvo | AUC médio | Skill acc. médio | Sep.econ média | Estab. (AUC± médio) | Horizontes significativos |
|---:|---|---:|---:|---:|---:|---:|
| 1 | volatilidade | 0.773 | +12.2% | 0.061% | 0.022 | 2/3 |
| 2 | mov_absoluto | 0.640 | +3.7% | 0.058% | 0.025 | 2/3 |
| 3 | regime | 0.634 | +10.9% | 0.103% | 0.022 | 2/3 |
| 4 | magnitude | 0.572 | +2.7% | 0.069% | 0.016 | 2/3 |
| 5 | direcao | 0.499 | +0.3% | 0.011% | 0.018 | 1/3 |

![Previsibilidade por alvo](target_predictability.png)

![Valor econômico por alvo](target_economic_value.png)

## 3. Conclusão objetiva

**Pergunta:** qual variável contém mais informação previsível do que simplesmente prever alta ou baixa?

**Resposta: `volatilidade` — Volatilidade futura (alta/baixa).**

- AUC médio de **0.773** vs. **0.499** da direção (acaso = 0,500). A direção permanece praticamente indistinguível do acaso; `volatilidade` carrega informação previsível real e estável (significativo em 2/3 horizontes).
- Alvos que superam a direção de forma significativa e estável: `volatilidade` (AUC 0.773), `mov_absoluto` (AUC 0.640), `regime` (AUC 0.634), `magnitude` (AUC 0.572).

**Implicação econômica:** a separação do |movimento| entre as classes previstas do melhor alvo é de ~0.061 pontos %, a ser comparada com o custo de ida-e-volta. Mesmo sem prever direção, antecipar o **tamanho/volatilidade** do movimento é a informação monetizável (sizing, opções, gestão de risco).

**Recomendação de pesquisa:** a direção binária está esgotada; o próximo alvo a modelar é **volatilidade** (Volatilidade futura (alta/baixa)). Só então faz sentido discutir features dedicadas — este estudo já o identifica como mais previsível **mesmo usando as features atuais, pensadas para direção**.

_Ressalva: previsibilidade medida com o conjunto de features atual e um único período. Um alvo promissor aqui é candidato a aprofundamento, não uma garantia de lucro — que dependerá de custos e execução._
