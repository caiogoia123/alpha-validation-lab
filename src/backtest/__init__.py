"""Suíte de validação científica: backtesting realista do sistema atual.

Não introduz novos indicadores, modelos ou fontes de dados. Reutiliza o mesmo
XGBoost e os mesmos indicadores já implementados, com o único objetivo de medir
se o sistema possui vantagem estatística real fora da amostra de treino.
"""
