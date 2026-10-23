import os
# Prevenção de vazamento de memória no Windows
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_DISPONIVEL = True
except ImportError:
    MLXTEND_DISPONIVEL = False

# ==============================================================================
# 1. LEITURA E PRÉ-PROCESSAMENTO (PANDAS & STANDARDSCALER)
# ==============================================================================
print("--- 1. Carregando e Padronizando Dados do IBGE ---")
df = pd.read_csv('ibge.csv')

atributos = ['Industria_Geral', 'Bens_Consumo', 'Bens_Capital', 'Intermediarios']
X = df[atributos].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# 2. AVALIAÇÃO DE K: MÉTODO DO COTOVELO (INÉRCIA) E COEFICIENTE DE SILHUETA
# ==============================================================================
inerciais = []
silhuetas = []
K = range(2, 11)

print("\n--- Testando K de 2 a 10 ---")
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inerciais.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, labels)
    silhuetas.append(score)
    
    print(f"K={k} | Inércia (Cotovelo): {kmeans.inertia_:.2f} | Silhueta: {score:.4f}")

# Gráficos Comparativos: Cotovelo vs Silhueta
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico do Cotovelo
ax1.plot(K, inerciais, marker='o', color='b', linewidth=2)
ax1.set_xlabel('Número de Clusters (K)')
ax1.set_ylabel('Inércia / WCSS')
ax1.set_title('Método do Cotovelo (Elbow Method)')
ax1.grid(True, linestyle='--')

# Gráfico da Silhueta
ax2.plot(K, silhuetas, marker='o', color='g', linewidth=2)
ax2.set_xlabel('Número de Clusters (K)')
ax2.set_ylabel('Coeficiente de Silhueta')
ax2.set_title('Coeficiente de Silhueta por K')
ax2.grid(True, linestyle='--')

plt.tight_layout()
plt.show()

# Seleção automática do melhor K pela Silhueta
best_k = K[silhuetas.index(max(silhuetas))]
print(f"\n=> Número ótimo de clusters selecionado: K={best_k}")

# ==============================================================================
# 3. K-MEANS FINAL E PERFIL DOS GRUPOS
# ==============================================================================
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

print("\nMédia dos Atributos por Cluster:")
print(df.groupby('Cluster')[atributos].mean())

print("\nEstados pertencentes a cada Grupo:")
for c in range(best_k):
    estados = df[df['Cluster'] == c]['Estado'].tolist()
    print(f"Cluster {c} ({len(estados)} estados): {', '.join(sorted(estados))}")

# ==============================================================================
# 4. REGRAS DE ASSOCIAÇÃO (APRIORI)
# ==============================================================================
print("\n--- Regras de Associação ---")
df_disc = pd.DataFrame()
for col in atributos:
    df_disc[f'{col}_Alto'] = df[col] >= 100.0

df_disc = df_disc.astype(str)
df_encoded = pd.get_dummies(df_disc)

if MLXTEND_DISPONIVEL:
    frequent_itemsets = apriori(df_encoded, min_support=0.4, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
    
    rules_true = rules[rules['consequents'].astype(str).str.contains('True')]
    
    print("\n🔹 Regras com Maior Confiança e Lift:\n")
    for idx, row in rules_true.head(5).iterrows():
        antecedente = ', '.join(list(row['antecedents']))
        consequente = ', '.join(list(row['consequents']))
        print(f"➡️ SE ({antecedente}) ENTÃO ({consequente})")
        print(f"   Suporte: {row['support']:.2f} | Confiança: {row['confidence']:.2f} | Lift: {row['lift']:.2f}\n")