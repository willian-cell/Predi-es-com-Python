import matplotlib.pyplot as plt
import pandas as pd

# Carregar o dataset
file_path = 'cancer_dataset.csv'
df = pd.read_csv(file_path)

# Exibir as primeiras linhas para entender a estrutura dos dados
df.head()


# Contar a quantidade de cada tipo de câncer
cancer_counts = df['Tipo de Câncer'].value_counts()

# Plotar gráfico de barras para os tipos de câncer
plt.figure(figsize=(10, 6))
plt.bar(cancer_counts.index, cancer_counts.values, color='teal')
plt.title('Distribuição dos Tipos de Câncer', fontsize=16)
plt.xlabel('Tipo de Câncer', fontsize=12)
plt.ylabel('Número de Pacientes', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Contagem por sexo
sex_counts = df['Sexo'].value_counts()

# Plotar gráfico de barras para a distribuição por sexo
plt.figure(figsize=(8, 5))
plt.bar(sex_counts.index, sex_counts.values, color='orange')
plt.title('Distribuição por Sexo', fontsize=16)
plt.xlabel('Sexo', fontsize=12)
plt.ylabel('Número de Pacientes', fontsize=12)
plt.tight_layout()
plt.show()

# Contagem por histórico familiar
history_counts = df['Histórico Familiar'].value_counts()

# Plotar gráfico de barras para a distribuição por histórico familiar
plt.figure(figsize=(8, 5))
plt.bar(history_counts.index, history_counts.values, color='purple')
plt.title('Distribuição por Histórico Familiar', fontsize=16)
plt.xlabel('Histórico Familiar', fontsize=12)
plt.ylabel('Número de Pacientes', fontsize=12)
plt.tight_layout()
plt.show()

# Contagem por tabagismo
smoking_counts = df['Tabagismo'].value_counts()

# Plotar gráfico de barras para a distribuição por tabagismo
plt.figure(figsize=(8, 5))
plt.bar(smoking_counts.index, smoking_counts.values, color='green')
plt.title('Distribuição por Tabagismo', fontsize=16)
plt.xlabel('Tabagismo', fontsize=12)
plt.ylabel('Número de Pacientes', fontsize=12)
plt.tight_layout()
plt.show()

