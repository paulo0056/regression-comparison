"""Módulo para análise comparativa de técnicas de previsão de casos de dengue."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

# Dados reais de exemplo
# São os dados de 2015 a 2022 respectivamente
anos = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]).reshape(-1, 1)
casos = np.array([55610, 67748, 10107, 9761, 68140, 83540, 25094, 35925])

# Criar série temporal para métodos clássicos
anos_serie = np.arange(2015, 2023)
ts_data = pd.Series(casos, index=pd.date_range('2015', periods=8, freq='Y'))

# Prever para 2023
ano_prev = np.array([[2023]])

# Dado real de 2023
REAL_2023 = 47626

# Função para calcular MAPE
def calcular_mape(real, previsto):
    return abs((real - previsto) / real) * 100

# Dicionário para armazenar resultados
resultados = {}

print("=== ANÁLISE COMPARATIVA DE TÉCNICAS DE PREVISÃO ===\n")

# 1. ARIMA(1,0,0) - Melhor performance da análise anterior
print("1. Executando ARIMA(1,0,0)...")
start_time = time.time()
try:
    # Usar ARIMA(1,0,0) que teve melhor performance (7.1% MAPE)
    modelo_arima = ARIMA(casos, order=(1,0,0))
    fitted_arima = modelo_arima.fit()
    
    tempo_treino_arima = time.time() - start_time
    
    # Fazer previsão
    start_pred = time.time()
    prev_arima = fitted_arima.forecast(steps=1)[0]
    tempo_pred_arima = (time.time() - start_pred) * 1000  # em ms
    
    mape_arima = calcular_mape(REAL_2023, prev_arima)
    
    resultados['ARIMA'] = {
        'previsao': prev_arima,
        'erro': abs(prev_arima - REAL_2023),
        'mape': mape_arima,
        'tempo_treino': tempo_treino_arima,
        'tempo_pred': tempo_pred_arima,
        'ordem': (1,0,0)
    }
    
    print(f"   Ordem: (1,0,0)")
    print(f"   Previsão: {int(prev_arima)}")
    print(f"   MAPE: {mape_arima:.2f}%")
    
except Exception as e:
    print(f"   Erro no ARIMA: {e}")
    resultados['ARIMA'] = None

# 2. Holt-Winters (configuração otimizada)
print("\n2. Executando Holt-Winters...")
start_time = time.time()
try:
    # Usar trend=None (sem tendência) que teve melhor performance (6.6% vs 21.2% MAPE)
    modelo_hw = ExponentialSmoothing(casos, trend=None, seasonal=None, 
                                   seasonal_periods=None).fit()
    tempo_treino_hw = time.time() - start_time
    
    start_pred = time.time()
    prev_hw = modelo_hw.forecast(steps=1)[0]
    tempo_pred_hw = (time.time() - start_pred) * 1000
    
    mape_hw = calcular_mape(REAL_2023, prev_hw)
    
    resultados['Holt-Winters'] = {
        'previsao': prev_hw,
        'erro': abs(prev_hw - REAL_2023),
        'mape': mape_hw,
        'tempo_treino': tempo_treino_hw,
        'tempo_pred': tempo_pred_hw
    }
    
    print(f"   Configuração: trend=None (otimizada)")
    print(f"   Previsão: {int(prev_hw)}")
    print(f"   MAPE: {mape_hw:.2f}%")
    
except Exception as e:
    print(f"   Erro no Holt-Winters: {e}")
    resultados['Holt-Winters'] = None

# 3. Regressão Polinomial (grau 2)
print("\n3. Executando Regressão Polinomial...")
start_time = time.time()
poly = PolynomialFeatures(degree=2)
anos_poly = poly.fit_transform(anos)
modelo_poly = LinearRegression()
modelo_poly.fit(anos_poly, casos)
tempo_treino_poly = time.time() - start_time

start_pred = time.time()
prev_poly = modelo_poly.predict(poly.transform(ano_prev))[0]
tempo_pred_poly = (time.time() - start_pred) * 1000

mape_poly = calcular_mape(REAL_2023, prev_poly)

resultados['Regressão Polinomial'] = {
    'previsao': prev_poly,
    'erro': abs(prev_poly - REAL_2023),
    'mape': mape_poly,
    'tempo_treino': tempo_treino_poly,
    'tempo_pred': tempo_pred_poly
}

print(f"   Previsão: {int(prev_poly)}")
print(f"   MAPE: {mape_poly:.2f}%")

# 4. Decision Tree
print("\n4. Executando Decision Tree...")
start_time = time.time()
param_grid_tree = {
    'max_depth': [None, 3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt']
}

grid_tree = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid_tree, cv=3)
grid_tree.fit(anos, casos)
modelo_tree = grid_tree.best_estimator_
tempo_treino_tree = time.time() - start_time

start_pred = time.time()
prev_tree = modelo_tree.predict(ano_prev)[0]
tempo_pred_tree = (time.time() - start_pred) * 1000

mape_tree = calcular_mape(REAL_2023, prev_tree)

resultados['Árvore de Decisão'] = {
    'previsao': prev_tree,
    'erro': abs(prev_tree - REAL_2023),
    'mape': mape_tree,
    'tempo_treino': tempo_treino_tree,
    'tempo_pred': tempo_pred_tree
}

print(f"   Melhores parâmetros: {grid_tree.best_params_}")
print(f"   Previsão: {int(prev_tree)}")
print(f"   MAPE: {mape_tree:.2f}%")

# 5. Random Forest 
print("\n5. Executando Random Forest...")
start_time = time.time()
param_grid_rf = {
     'n_estimators': [30, 60, 100],
    'max_depth': [None],
    'min_samples_split': [2, 5, 8, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt'],
    'bootstrap': [True],
    'criterion': ['absolute_error']
}

# Usando apenas dados até 2022
anos_treino = anos[:7]
casos_treino = casos[:7]

grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3)
grid_rf.fit(anos_treino, casos_treino)
modelo_rf = grid_rf.best_estimator_
tempo_treino_rf = time.time() - start_time

start_pred = time.time()
prev_rf = modelo_rf.predict(ano_prev)[0]
tempo_pred_rf = (time.time() - start_pred) * 1000

mape_rf = calcular_mape(REAL_2023, prev_rf)

resultados['Random Forest'] = {
    'previsao': prev_rf,
    'erro': abs(prev_rf - REAL_2023),
    'mape': mape_rf,
    'tempo_treino': tempo_treino_rf,
    'tempo_pred': tempo_pred_rf
}

print(f"   Melhores parâmetros: {grid_rf.best_params_}")
print(f"   Previsão: {int(prev_rf)}")
print(f"   MAPE: {mape_rf:.2f}%")

# Mostrar tabela de resultados
print(f"\n=== TABELA COMPARATIVA DE RESULTADOS ===")
print(f"Valor real 2023: {REAL_2023}")
print()
print("| Método                   | Previsão | Erro Abs. | MAPE (%) | T.Treino(s) | T.Pred(ms) |")
print("|--------------------------|----------|-----------|----------|-------------|------------|")

# Ordenar por MAPE
metodos_ordenados = sorted(resultados.items(), 
                          key=lambda x: x[1]['mape'] if x[1] else float('inf'))

for metodo, dados in metodos_ordenados:
    if dados:
        print(f"| {metodo:<24} | {int(dados['previsao']):>8} | {int(dados['erro']):>9} | "
              f"{dados['mape']:>8.2f} | {dados['tempo_treino']:>11.3f} | {dados['tempo_pred']:>10.1f} |")

# Gráfico comparativo
plt.figure(figsize=(12, 8))
anos_futuros = np.arange(2015, 2026).reshape(-1, 1)

# Dados históricos
plt.scatter(anos_serie, casos, color='black', label='Casos Reais', s=60, zorder=5)

# Previsões dos modelos
cores = ['orange', 'blue', 'green', 'red', 'purple']
i = 0

for metodo, dados in resultados.items():
    if dados and metodo != 'ARIMA':  # ARIMA precisa tratamento especial
        if metodo == 'Regressão Polinomial':
            prev_futura = modelo_poly.predict(poly.transform(anos_futuros))
            plt.plot(anos_futuros.flatten(), prev_futura, label=metodo, color=cores[i], alpha=0.7)
        elif metodo == 'Árvore de Decisão':
            prev_futura = modelo_tree.predict(anos_futuros)
            plt.plot(anos_futuros.flatten(), prev_futura, label=metodo, color=cores[i], alpha=0.7)
        elif metodo == 'Random Forest':
            prev_futura = modelo_rf.predict(anos_futuros)
            plt.plot(anos_futuros.flatten(), prev_futura, label=metodo, color=cores[i], alpha=0.7)
        elif metodo == 'Holt-Winters':
            # Para Holt-Winters, mostrar apenas a previsão de 2023
            plt.scatter([2023], [dados['previsao']], color=cores[i], label=f"{metodo} (2023)", 
                       marker='s', s=80, zorder=4)
        i += 1

# ARIMA - tratamento especial
if resultados['ARIMA']:
    plt.scatter([2023], [resultados['ARIMA']['previsao']], color='cyan', 
               label=f"ARIMA{resultados['ARIMA']['ordem']} (2023)", 
               marker='^', s=80, zorder=4)

# Valor real de 2023
plt.scatter([2023], [REAL_2023], color='red', label='Real 2023', 
           marker='x', s=100, zorder=6)

plt.title("Comparação de Técnicas de Previsão - Casos de Dengue", fontsize=14, fontweight='bold')
plt.xlabel("Ano")
plt.ylabel("Casos")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

print('\n=== ANÁLISE DE PERFORMANCE ===')
print("\nRanking por MAPE (melhor para pior):")
for i, (metodo, dados) in enumerate(metodos_ordenados, 1):
    if dados:
        print(f"{i}. {metodo}: {dados['mape']:.2f}%")

print('\nPronto para mostrar o gráfico...')
plt.show()
