
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# Verificar se o arquivo do dataset existe e carregá-lo
try:
    df = pd.read_csv("cancer_dataset.csv")
except FileNotFoundError:
    raise FileNotFoundError("O arquivo 'cancer_dataset.csv' não foi encontrado. Certifique-se de que o caminho está correto.")

# Configuração do pipeline de pré-processamento
variaveis_numericas = ["Idade"]  # Adicionar mais variáveis numéricas, se necessário
variaveis_categoricas = ["Sexo", "Histórico Familiar", "Tabagismo", "Comorbidades", "Estágio Diagnóstico"]

# Pipeline de pré-processamento
transformador_numerico = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="mean")),
    ("padronizador", StandardScaler())
])

transformador_categorico = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="most_frequent")),
    ("codificador", OneHotEncoder(handle_unknown="ignore"))
])

# Combinar transformadores no ColumnTransformer
preprocessador = ColumnTransformer(
    transformers=[
        ("numerico", transformador_numerico, variaveis_numericas),
        ("categorico", transformador_categorico, variaveis_categoricas)
    ]
)

# Pipeline completo com pré-processamento e modelo RandomForest
pipeline_modelo = Pipeline(steps=[
    ("preprocessador", preprocessador),
    ("classificador", RandomForestClassifier(random_state=42))
])

# Verificar se as colunas necessárias existem no DataFrame
colunas_necessarias = ["Tipo de Câncer"] + variaveis_numericas + variaveis_categoricas
for coluna in colunas_necessarias:
    if coluna not in df.columns:
        raise ValueError(f"A coluna '{coluna}' não foi encontrada no DataFrame. Certifique-se de que todas as colunas necessárias estão presentes.")

# Criar a variável alvo e preparar os dados
df['Cancer_Presente'] = np.where(df['Tipo de Câncer'].notnull(), 1, 0)

X = df.drop(columns=["Tipo de Câncer", "Cancer_Presente"])
y = df["Cancer_Presente"]

# Dividir os dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# Balancear os dados, caso estejam desbalanceados
if y_treino.value_counts().min() == 0:
    # Combinar dados de treino e variável alvo
    dados_treino = pd.concat([X_treino, y_treino], axis=1)

    # Dividir entre classes majoritária e minoritária
    classe_majoritaria = dados_treino[dados_treino["Cancer_Presente"] == 1]
    classe_minoritaria = dados_treino[dados_treino["Cancer_Presente"] == 0]

    # Realizar upsampling da classe minoritária
    classe_minoritaria_ampliada = resample(classe_minoritaria, replace=True, n_samples=len(classe_majoritaria), random_state=42)

    # Combinar novamente as classes
    dados_treino_balanceados = pd.concat([classe_majoritaria, classe_minoritaria_ampliada])

    # Separar novamente variáveis independentes e alvo
    X_treino_balanceado = dados_treino_balanceados.drop(columns=["Cancer_Presente"])
    y_treino_balanceado = dados_treino_balanceados["Cancer_Presente"]
else:
    X_treino_balanceado = X_treino
    y_treino_balanceado = y_treino

# Treinar o modelo com os dados balanceados ou não
pipeline_modelo.fit(X_treino_balanceado, y_treino_balanceado)

# Obter as probabilidades preditas para o conjunto de teste
probabilidades_preditas = pipeline_modelo.predict_proba(X_teste)

# Extrair a probabilidade da classe negativa (Câncer Ausente)
probabilidade_cancer = probabilidades_preditas[:, 0]

# Criar DataFrame com as probabilidades de predisposição ao câncer
df_probabilidades = pd.DataFrame(probabilidade_cancer, columns=["Probabilidade_Cancer"], index=X_teste.index)

# Selecionar as 10 pessoas mais predispostas ao câncer
top_predispostos_cancer = df_probabilidades.nlargest(10, "Probabilidade_Cancer")

# Exibir as 10 pessoas mais predispostas
print(top_predispostos_cancer)

# Gerar gráfico de barras para visualizar as probabilidades
plt.figure(figsize=(10, 6))
plt.bar(top_predispostos_cancer.index, top_predispostos_cancer["Probabilidade_Cancer"], color='darkred')
plt.title("Top 10 Pacientes com Maior Predisposição ao Câncer", fontsize=16)
plt.xlabel("Índice do Paciente", fontsize=12)
plt.ylabel("Probabilidade de Câncer", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


