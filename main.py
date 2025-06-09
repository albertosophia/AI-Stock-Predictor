'''

GRUPO:

ALBERTO SOPHIA NETO
LUCAS BEATO
LUIS FERNANDO GERONIMO
MATHEUS AROEIRA
MATHEUS GASPARINI

'''



'''

TRABALHO MATERIA INTELIGENCIA ARTIFICIAL:

REDE NEURAL CAPAZ DE PREVER QUEDA OU ALTA DA BOLSA DE VALORES.

MODELO PARA FINS EDUCACIONAIS, NÃO INDICADA PARA USO FINANCEIRO.



PELO QUE TESTAMOS A PRECISÃO DELE NÃO ESTA RUIM, MAS LONGE DE SER ALGO VIAVEL PARA UTILIZAR DIARIAMENTE.

A MAIOR PRECISAO DELE NAO ESTA NOS VALORES E SIM NA PREVISAO DE ALTA E QUEDA, ESTA COM UMA DIFERENCA ALTA DE VALORES.

GERALMENTE SE ELE PREVE QUE UMA BOLSA AMANHA VAI ESTAR A 30 REAIS E VAI ESTAR EM ALTA, ELE ACERTA QUE ESTARA EM ALTA, MAS ESSE VALOR DE 30 REAIS OU
ELE ERRA PARA CIMA OU ERRA PARA BAIXO, ENTAO SE ELE PREVIU 30 REAIS, AMANHA POSSA SER QUE A BOLSA VA PARA 35 OU 25 (VALOR DE EXEMPLO).

COMO FOI ALGO PARA INTRODUZIR NOS ALUNOS AS REDES NEURAIS, EU DIRIA QUE NOSSA REDE ESTA OK. CLARO QUE LONGE DE SER ALGO UTILIZAVEL.


'''





import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#Pegar o símbolo da ação digitado pelo usuário
stock_symbol = input("Digite o símbolo da ação (ex: PETR4.SA para Petrobras): ")

#Baixar os dados da ação (últimos 5 anos)
data = yf.download(stock_symbol, period="5y")
if data.empty:
    print(f"Erro: Não foi possível baixar dados para {stock_symbol}. Verifique o símbolo e tente novamente.")
    exit()

#Usar apenas o preço de fechamento ('Close')
prices = data['Close'].values.reshape(-1, 1)

#Normalizar os dados
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

lookback = 60
X, y = [], []
for i in range(lookback, len(prices_scaled)):
    X.append(prices_scaled[i - lookback:i, 0])
    y.append(prices_scaled[i, 0])
X, y = np.array(X), np.array(y)


X = np.reshape(X, (X.shape[0], X.shape[1], 1))

#Dividir em conjuntos de treino e teste (80% treino, 20% teste)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#Construir o modelo
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

#Treinar o modelo
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

#Fazer previsões no conjunto de teste
predicted_scaled = model.predict(X_test)
predicted = scaler.inverse_transform(predicted_scaled)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

#Avaliar o modelo com o Erro Quadrático Médio (MSE)
mse = mean_squared_error(y_test_real, predicted)
print(f"Erro Quadrático Médio (MSE): {mse:.4f}")

#Obter o preço atual e prever o próximo preço
#Preço atual: último preço de fechamento nos dados
current_price = data['Close'].iloc[-1]

#Prever o próximo preço (usando os últimos 60 dias)
last_sequence = prices_scaled[-lookback:].reshape(1, lookback, 1)
next_price_scaled = model.predict(last_sequence)
next_price = scaler.inverse_transform(next_price_scaled)[0][0]

#Print do preço atual e do preço previsto
print(f"Preço atual de {stock_symbol} (último fechamento): R${current_price.iloc[0]:.2f}")
print(f"Preço previsto para o próximo dia: R${next_price:.2f}")

if next_price > current_price.iloc[0]:
    print("Previsão de SUBIR!")
else:
    print("Previsão de CAIR!")

#Plotar os resultados
plt.figure(figsize=(10, 6))
plt.plot(y_test_real, label='Preço Real')
plt.plot(predicted, label='Preço Previsto')
plt.title(f'Previsão de Preços para {stock_symbol}')
plt.xlabel('Tempo')
plt.ylabel('Preço')
plt.legend()
plt.show()