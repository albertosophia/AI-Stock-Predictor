# Preditor de Tendência da Bolsa de Valores

## Visão Geral

Este projeto apresenta uma Rede Neural Recorrente (RNN) do tipo Long Short-Term Memory (LSTM) desenvolvida para **prever a tendência (alta ou queda)** de ações na bolsa de valores. O modelo foi criado como um **trabalho acadêmico** para a disciplina de Inteligência Artificial na **Unifadra - Fundec**, servindo como uma **ferramenta educacional** para estudantes, oferecendo uma introdução prática ao uso de redes neurais em séries temporais financeiras.

**É fundamental ressaltar que este modelo não se destina a uso financeiro real ou tomada de decisões de investimento.**

---

## Funcionalidades

* **Previsão de Tendência:** O modelo é otimizado para prever se o preço de uma ação irá subir ou cair no próximo dia de negociação.
* **Visualização de Resultados:** Apresenta gráficos comparando os preços reais com os preços previstos no conjunto de teste.
* **Fácil de Usar:** Solicita ao usuário o símbolo da ação (ticker) para realizar a análise.

---

## Como Funciona

O modelo utiliza dados históricos de preços de fechamento de ações, baixados diretamente do Yahoo Finance, para treinar uma rede neural LSTM. A arquitetura da rede é projetada para capturar padrões sequenciais nos dados, o que é crucial para a previsão de séries temporais.

### Etapas Principais:

1.  **Coleta de Dados:** Baixa os últimos 5 anos de dados históricos de fechamento para a ação especificada.
2.  **Pré-processamento:** Os dados são normalizados usando `MinMaxScaler` para otimizar o treinamento da rede neural. São criadas sequências de `lookback` (60 dias) para alimentar o modelo.
3.  **Divisão Treino/Teste:** Os dados são divididos em conjuntos de treino (80%) e teste (20%) para avaliar o desempenho do modelo em dados não vistos.
4.  **Construção e Treinamento do Modelo:** Uma rede LSTM com camadas `Dropout` é construída e treinada para aprender os padrões nos dados históricos.
5.  **Previsão:** O modelo prevê o preço de fechamento do próximo dia e, com base nisso, determina a tendência (alta ou queda) em relação ao preço atual.
6.  **Avaliação e Visualização:** O desempenho do modelo é avaliado usando o Erro Quadrático Médio (MSE), e os resultados são plotados para uma análise visual.

---

## Instalação e Uso

### Pré-requisitos

Certifique-se de ter as seguintes bibliotecas Python instaladas:

* `yfinance`
* `numpy`
* `pandas`
* `scikit-learn`
* `tensorflow` (ou `keras`)
* `matplotlib`

Você pode instalá-las via pip:


pip install yfinance numpy pandas scikit-learn tensorflow matplotlib


### Executando o Código

1.  Clone este repositório (ou copie o código `main.py`).
2.  Navegue até o diretório do projeto no seu terminal.
3.  Execute o script Python:

    
    python main.py
    

4.  Quando solicitado, digite o símbolo da ação (ticker) que deseja analisar. Exemplos:
    * `PETR4.SA` para Petrobras (ações brasileiras)
    * `AAPL` para Apple (ações americanas)

---

## Considerações e Limitações

Conforme mencionado, este modelo é para **fins educacionais** e demonstra os conceitos básicos de redes neurais para previsão de séries temporais. É importante ter em mente as seguintes limitações:

* **Precisão de Valores:** Embora o modelo demonstre uma capacidade razoável de prever a **tendência** (alta ou queda), a **precisão dos valores exatos** dos preços previstos é limitada. Frequentemente, há uma diferença significativa entre o preço previsto e o preço real, mesmo quando a direção da tendência é acertada.
* **Modelo Simplificado:** Para fins de introdução, a arquitetura do modelo é relativamente simples e pode não capturar todas as complexidades do mercado de ações.
* **Fatores Externos:** Modelos financeiros reais incorporam uma vasta gama de dados e fatores externos (notícias, indicadores macroeconômicos, volume de negociação, etc.) que não são considerados neste modelo educacional.
* **Volatilidade do Mercado:** O mercado de ações é inerentemente volátil e influenciado por inúmeros fatores imprevisíveis, tornando a previsão precisa um desafio extremamente complexo.

---

## Grupo

* ALBERTO SOPHIA NETO
* LUCAS BEATO
* LUIS FERNANDO GERONIMO
* MATHEUS AROEIRA
* MATHEUS GASPARINI

---