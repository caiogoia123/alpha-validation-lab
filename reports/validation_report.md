# Validação Anti-Overfitting — CV Purgada e Deflated Sharpe

_Gerado em 2026-06-01 21:48:17 · H=15 · 36,000 candles de teste._

> Duas defesas que separam um achado real de um artefato de busca: validação cruzada **purgada e com embargo** (López de Prado) e o **Deflated Sharpe Ratio** (Bailey & López de Prado).

## 1. Volatilidade sob CV purgada vs. baseline trivial

O baseline de persistência prevê 'alta vol futura' a partir da vol realizada das últimas H barras (causal, sem treino). Se o XGBoost não o superar, o 'achado' seria apenas clustering de volatilidade redescoberto.

| | AUC médio (folds) | Desvio | Folds |
|---|---:|---:|---:|
| **XGBoost (modelo)** | 0.743 | 0.048 | 5 |
| Baseline persistência | 0.812 | 0.013 | 5 |
| **Ganho do modelo** | **-0.070** | | |

**Leitura:** sob CV purgada o AUC do modelo é 0.743 vs. 0.812 do baseline; **o baseline trivial VENCE o modelo** — o XGBoost com features de TA é uma forma *pior* de capturar o clustering de volatilidade do que simplesmente usar a vol recente. A vol é genuinamente previsível (ambos ≫ 0,5), mas o 'achado' de ML não passa de persistência: a celebrada AUC ~0,79 **não sobrevive como mérito do modelo** quando confrontada com uma linha de baseline e CV sem vazamento.

## 2. Deflated Sharpe Ratio sobre as estratégias testadas

Foram testadas **12 configurações** de estratégia (A/B/C × {sem filtro, top 10/20/30%}). Reportar só a melhor infla o Sharpe por viés de seleção; o DSR corrige isso.

| Configuração | Sharpe/op | Nº ops |
|---|---:|---:|
| C_momentum / top 10% ⭐ | -0.5285 | 306 |
| B_breakout / top 10% | -0.5467 | 138 |
| A_ema / top 10% | -0.5472 | 257 |
| C_momentum / top 20% | -0.6088 | 562 |
| B_breakout / top 20% | -0.6141 | 223 |
| A_ema / top 20% | -0.6355 | 471 |
| C_momentum / top 30% | -0.6442 | 815 |
| A_ema / top 30% | -0.6877 | 658 |
| B_breakout / top 30% | -0.7082 | 298 |
| C_momentum / sem filtro | -0.7859 | 1,799 |
| A_ema / sem filtro | -0.8591 | 1,443 |
| B_breakout / sem filtro | -0.9053 | 833 |

| Métrica | Valor |
|---|---|
| Melhor configuração | C_momentum / top 10% |
| Sharpe observado (melhor) | -0.5285 |
| Nº de tentativas | 12 |
| Benchmark deflacionado (E[máx] sob H0) | +0.2041 |
| PSR vs. 0 | 0.000 |
| **Deflated Sharpe Ratio** | **0.000** |

**Leitura:** o DSR é a probabilidade de o Sharpe da melhor estratégia ser real (não fruto de testar muitas). Aqui DSR = **0.000** (≪ 0,95). **Não há skill estatístico após a deflação** — o melhor Sharpe é compatível com o esperado ao testar tantas configurações sem edge real.

## 3. Conclusão

- A volatilidade é previsível, mas isso é **persistência trivial**: sob CV purgada, o baseline de vol recente (0.812) **supera** o XGBoost (0.743); ganho do modelo = -0.070 de AUC. O mérito de ML do 'achado' original não se sustenta — era clustering de volatilidade.
- Nenhuma estratégia direcional simples sobrevive ao **Deflated Sharpe**: o melhor resultado é estatisticamente indistinguível de ruído de seleção.
- Conclusão honesta reforçada: **há sinal de volatilidade, não há edge direcional** — nem mesmo após explorar o sinal de vol como filtro.
