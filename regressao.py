import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dados de 2015 até 2023


anos = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]).reshape(-1, 1)
casos = np.array([55610, 67748, 10107, 9761, 68140, 83540, 25094, 35925])

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(anos, casos)

# Prever 2024
previsao_2023 = modelo.predict([[2024]])
print(f'Previsão para 2024: {int(previsao_2023[0])} casos')

# Comparar com o valor real
real_2023 = 47626
erro = real_2023 - previsao_2023[0]
print(f'Diferença entre real e previsto: {int(erro)} casos')

# (Opcional) Visualizar o gráfico
plt.scatter(anos, casos, color='blue', label='Casos reais')
plt.plot(range(2015, 2026), modelo.predict(np.array(range(2015, 2026)).reshape(-1, 1)), color='red', label='Regressão Linear')
plt.scatter([2023], [real_2023], color='green', label='Valor Real 2023')
plt.title("Previsão de Casos de Dengue")
plt.xlabel("Ano")
plt.ylabel("Casos")
plt.legend()
plt.grid(True)
plt.show()
