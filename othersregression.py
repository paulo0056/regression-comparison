import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#  Dados reais de exemplo 
anos = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]).reshape(-1, 1)
casos = np.array([55610, 67748, 10107, 9761, 68140, 83540, 25094, 35925])

#  Prever para 2024 
#ano_prev = np.array([[2024]])

#  Prever para 2023 
ano_prev = np.array([[2023]])
real_2023 = 47626
# 1. Regressão Linear
modelo_linear = LinearRegression()
modelo_linear.fit(anos, casos)
prev_linear = modelo_linear.predict(ano_prev)

# 2. Regressão Polinomial (grau 2)
poly = PolynomialFeatures(degree=2)
anos_poly = poly.fit_transform(anos)
modelo_poly = LinearRegression()
modelo_poly.fit(anos_poly, casos)
prev_poly = modelo_poly.predict(poly.transform(ano_prev))

# 3. Decision Tree
param_grid_tree = {
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1,2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

grid_tree = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid_tree, cv=3)
grid_tree.fit(anos, casos)
modelo_tree = grid_tree.best_estimator_
prev_tree = modelo_tree.predict(ano_prev)
print(f"Melhores parâmetros da Decision Tree: {grid_tree.best_params_}")

# 4. Random Forest (ajuste de hiperparâmetros refinado) SEM VIÉS
param_grid_rf = {
     'n_estimators': [30, 60, 100],
    'max_depth': [None],
    'min_samples_split': [2, 5, 8, 10],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt'],
    'bootstrap': [True],
    'criterion': ['absolute_error']
}

# Use apenas dados até 2022 para ajuste e treino
anos_treino = anos[:7]  # 2015 a 2022
casos_treino = casos[:7]

grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3)
grid_rf.fit(anos_treino, casos_treino)
modelo_rf = grid_rf.best_estimator_
prev_rf = modelo_rf.predict(ano_prev)
print(f"Melhores parâmetros do Random Forest: {grid_rf.best_params_}")

#  Mostrar previsões
#print("📊 Previsões para 2024:")
print(f"Valor real: {real_2023}")
print("📊 Previsões para 2023:")
print(f"Linear: {int(prev_linear[0])}")
print(f"Diferença entre real e previsto: {int(prev_linear[0] - real_2023)}")
print(f"Polinomial (grau 2): {int(prev_poly[0])}")
print(f"Diferença entre real e previsto: {int(prev_poly[0] - real_2023)}")
print(f"Decision Tree: {int(prev_tree[0])}")
print(f"Diferença entre real e previsto: {int(prev_tree[0] - real_2023)}")
print(f"Random Forest: {int(prev_rf[0])}")
print(f"Diferença entre real e previsto: {int(prev_rf[0] - real_2023)}")

#  Gráficos comparando
anos_futuros = np.arange(2015, 2026).reshape(-1, 1)
plt.scatter(anos, casos, color='black', label='Casos Reais')

plt.plot(anos_futuros, modelo_linear.predict(anos_futuros), label='Linear', color='blue')
plt.plot(anos_futuros, modelo_poly.predict(poly.transform(anos_futuros)), label='Polinomial (grau 2)', color='orange')
plt.plot(anos_futuros, modelo_tree.predict(anos_futuros), label='Decision Tree', color='green')
plt.plot(anos_futuros, modelo_rf.predict(anos_futuros), label='Random Forest', color='red')

# Dado real de 2023 


plt.scatter([2023], [real_2023], color='purple', label='Real 2023', marker='x', s=100)

plt.title("Comparação de Técnicas de Regressão - Casos de Dengue")
plt.xlabel("Ano")
plt.ylabel("Casos")
plt.legend()
plt.grid(True)
print('Pronto para mostrar o gráfico...')
plt.show()
