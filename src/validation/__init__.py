"""Validação anti-overfitting de nível quant.

Reúne três defesas que separam um achado real de um artefato de busca:
  * purged_cv     — K-fold purgado e com embargo para séries com rótulos que
                    olham o futuro (López de Prado), evitando vazamento entre
                    treino e teste por sobreposição temporal.
  * deflated_sharpe — Probabilistic e Deflated Sharpe Ratio (Bailey & López de
                    Prado), que descontam o viés de seleção por testar muitas
                    estratégias.
  * baselines     — baseline trivial de persistência de volatilidade, para
                    provar que o modelo agrega valor além de "vol futura = vol
                    recente".
"""
