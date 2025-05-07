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

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
    'criterion': ['squared_error', 'absolute_error']
}

grid_rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid_rf,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_absolute_error'
)
grid_rf.fit(anos, casos)  # anos e casos agora com 20 exemplos
print("Melhores parâmetros:", grid_rf.best_params_)

#  Mostrar previsões
#print("📊 Previsões para 2024:")
print(f"Valor real: {real_2023}")
print("📊 Previsões para 2023:")
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
