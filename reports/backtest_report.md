# Relatório de Validação — BTC/USDT (+5 min)

_Gerado em 2026-06-01 20:40:28 · out-of-sample de 2026-05-15 14:48 UTC a 2026-06-01 23:27 UTC._

> Validação científica do sistema **atual** (sem novos indicadores, modelos ou fontes de dados). O modelo é treinado **apenas no in-sample** e avaliado **out-of-sample**; todos os baselines passam pelo mesmo motor de simulação (mesmos SL/TP, taxa e slippage).

## 1. Configuração do backtest

| Parâmetro | Valor |
|---|---|
| Capital inicial | 10,000.00 USDT |
| Valor por operação | 1,000.00 USDT |
| Stop Loss | 0.300% |
| Take Profit | 0.300% |
| Taxa da corretora (por lado) | 0.040% |
| Slippage (por execução) | 0.020% |
| Confiança mínima (modelo) | 50% |
| Horizonte / holding máx. | 5 candles |
| Candles totais | 250000 (treino 225000 / teste 25000) |

## 2. Métricas de trading — Modelo vs Baselines

| Estratégia | Ops | Acerto dir. | Win rate líq. | Profit Factor | Expectância (USDT/op) | Sharpe/op | Retorno acum. | Drawdown máx. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Modelo (IA) | 4207 | 50.0% | 7.3% | 0.042 | -1.1979 | -1.289 | -50.40% | -50.39% |
| Aleatório (cara/coroa) | 4203 | 50.9% | 7.5% | 0.045 | -1.1863 | -1.284 | -49.86% | -49.85% |
| Sempre comprado | 4203 | 49.0% | 6.8% | 0.042 | -1.2126 | -1.310 | -50.96% | -50.96% |
| Sempre vendido | 4205 | 51.4% | 7.8% | 0.049 | -1.1776 | -1.266 | -49.52% | -49.51% |
| Cruzamento EMA9/21 | 4199 | 48.5% | 7.1% | 0.045 | -1.2096 | -1.315 | -50.79% | -50.78% |

> **Acerto dir.** = preço foi na direção prevista (skill do sinal, sem custos). **Win rate líq.** = operações lucrativas após taxa e slippage. A diferença entre as duas colunas é o quanto os custos corroem o sinal.

Detalhe de ganhos/perdas do modelo:

| Métrica | Valor |
|---|---|
| Ganho médio (operações vencedoras) | 0.7189 USDT |
| Perda média (operações perdedoras) | -1.3483 USDT |
| Lucro bruto / Prejuízo bruto | 219.97 / 5259.74 USDT |
| PnL líquido total | -5039.77 USDT |
| Saldo final | 4,960.23 USDT |
| Saídas (SL / TP / Tempo) | 105 / 49 / 4053 |

## 3. Análise por faixa de confiança (modelo)

Operações independentes por sinal (amostra cheia), para avaliar se convicção maior produz resultado melhor.

| Faixa | Operações | Acerto direcional | Win rate líq. | Lucro líquido (USDT) |
|---|---:|---:|---:|---:|
| 50-60% | 24712 | 51.1% | 6.7% | -29635.08 |
| 60-70% | 287 | 57.8% | 13.6% | -321.28 |
| 70-80% | 0 | — | — | 0.00 |
| 80-90% | 0 | — | — | 0.00 |
| 90-100% | 0 | — | — | 0.00 |

**Monotonicidade (Spearman faixa × acerto direcional):** rho = 1.00 — tendência positiva.

## 4. Curva de capital

![Curva de capital](equity_curve.png)

Saldo (azul), pico vigente (tracejado), drawdowns (vermelho) e lucro acumulado (verde, eixo à direita).

## 5. Conclusão objetiva

**Pergunta:** o modelo atual apresenta vantagem estatística suficiente para justificar mais desenvolvimento?

**Resposta: NÃO** (1/5 critérios atendidos).

| Critério | Resultado | Atende? |
|---|---|:---:|
| Expectância por operação positiva (após custos) | -1.1979 USDT/op | ❌ |
| Profit Factor > 1,10 | 0.042 | ❌ |
| Acurácia direcional > 52% (skill acima do acaso) | 50.0% | ❌ |
| Supera o melhor baseline em expectância (Sempre vendido) | modelo -1.1979 vs -1.1776 | ❌ |
| Confiança maior associada a maior acerto direcional (Spearman > 0,3) | rho = 1.00 | ✅ |

O sistema não demonstra vantagem estatística suficiente sobre baselines triviais após custos. Mais desenvolvimento no formato atual não se justifica.

_Ressalva metodológica: resultado de um único split temporal e de um período de mercado específico. Custos (taxa+slippage) penalizam fortemente estratégias de alta frequência — exatamente o efeito que este backtest busca expor._
