"""Validação econômica da previsibilidade de volatilidade.

Mede se a informação já descoberta (volatilidade futura é previsível, AUC ~0,77)
tem VALOR ECONÔMICO utilizável — sem novos indicadores, features ou modelos.
Reutiliza o mesmo XGBoost (classificador de volatilidade alta/baixa), os mesmos
indicadores e o mesmo motor de backtest das etapas anteriores.
"""
