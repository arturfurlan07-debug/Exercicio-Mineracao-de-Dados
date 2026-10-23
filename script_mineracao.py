import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Tentar importar mlxtend para regras de associação
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_DISPONIVEL = True
except ImportError:
    MLXTEND_DISPONIVEL = False

# ==============================================================================
# ETAPA 1: LEITURA DOS DADOS
# ==============================================================================
print("--- ETAPA 1: Carregando os Dados ---")
df = pd.read_csv('ibge.csv')

print(f"Total de registos: {df.shape[0]} estados")
print("\nPrimeiras linhas:")
print(df.head())

# ==============================================================================
# ETAPA 2: PRÉ-PROCESSAMENTO E PADRONIZAÇÃO (Z-SCORE)
# ==============================================================================
print("\n--- ETAPA 2: Pré-processamento dos Dados ---")

atributos = ['Industria_Geral', 'Bens_Consumo', 'Bens_Capital', 'Intermediarios']
X = df[atributos]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# ETAPA 3: CÁLCULO DO MELHOR K VIA COEFICIENTE DE SILHUETA
# ==============================================================================
print("\n--- ETAPA 3: Avaliação de Silhueta para Encontrar K Ideal ---")

silhuetas = {}
for k in range(2, 9):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    rotulos = kmeans_temp.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, rotulos)
    silhuetas[k] = score
    print(f"K = {k}: Coeficiente de Silhueta = {score:.4f}")

melhor_k = max(silhuetas, key=silhuetas.get)
print(f"\n=> O valor ideal de K baseado na maior Silhueta é K = {melhor_k}")

# ==============================================================================
# ETAPA 4: APLICAÇÃO DO ALGORITMO K-MEANS
# ==============================================================================
print(f"\n--- ETAPA 4: Aplicação do K-Means com K={melhor_k} ---")

kmeans = KMeans(n_clusters=melhor_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ==============================================================================
# ETAPA 5: APRESENTAÇÃO DOS GRUPOS ENCONTRADOS
# ==============================================================================
print("\n--- ETAPA 5: Perfil dos Grupos Encontrados ---")

medias_cluster = df.groupby('Cluster')[atributos].mean()
print("\nMédia das Variáveis por Cluster:")
print(medias_cluster)

print("\nEstados pertencentes a cada Grupo:")
for c in range(melhor_k):
    estados_grupo = df[df['Cluster'] == c]['Estado'].tolist()
    print(f"Cluster {c} ({len(estados_grupo)} estados): {', '.join(sorted(estados_grupo))}")

# ==============================================================================
# ETAPA 6: REGRAS DE ASSOCIAÇÃO
# ==============================================================================
print("\n--- ETAPA 6: Regras de Associação ---")

df_binario = pd.DataFrame()
for col in atributos:
    df_binario[f'{col}_Alto'] = df[col] >= 100.0

if MLXTEND_DISPONIVEL:
    itemsets_frequentes = apriori(df_binario, min_support=0.4, use_colnames=True)
    regras = association_rules(itemsets_frequentes, metric="confidence", min_threshold=0.8)
    regras_ordenadas = regras.sort_values(by=['lift', 'confidence'], ascending=False)
    
    print("\nPrincipais Regras de Associação Encontradas:")
    for idx, row in regras_ordenadas.head(5).iterrows():
        ant = list(row['antecedents'])
        cons = list(row['consequents'])
        print(f"SE {ant} ENTÃO {cons} | Suporte: {row['support']:.2f} | Confiança: {row['confidence']:.2f} | Lift: {row['lift']:.2f}")
else:
    print("\n[Aviso] Instale a biblioteca 'mlxtend' para ver as regras de associação no terminal.")