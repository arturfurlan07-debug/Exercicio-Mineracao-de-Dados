import pandas as pd
import numpy as np

# Tentar importar o mlxtend para mineração Apriori
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_DISPONIVEL = True
except ImportError:
    MLXTEND_DISPONIVEL = False

# ==============================================================================
# ETAPA 1: LEITURA DA BASE DE DADOS
# ==============================================================================
print("--- ETAPA 1: Leitura da Base de Dados ---")
url = "https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv"

# Carregamento do dataset diretamente da URL
df = pd.read_csv(url)

print(f"Total de registos: {df.shape[0]} pacientes")
print(f"Colunas presentes: {list(df.columns)}")
print("\nPrimeiras 5 linhas do dataset:")
print(df.head())

# Remoção da coluna 'Id' ou colunas de índice não informativas
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# ==============================================================================
# ETAPA 2: TRATAMENTO E TRANSFORMAÇÃO DE VARIÁVEIS EM CATEGORIAS
# ==============================================================================
print("\n--- ETAPA 2: Transformação de Variáveis em Categorias ---")

# Tratamento de valores faltantes (ex: 'Bare.nuclei' contendo NAs)
for col in df.columns:
    if col != 'Class':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Preenchimento de NAs pela mediana
df = df.fillna(df.median(numeric_only=True))

# As características médicas (1 a 10) são discretizadas em intervalos:
# 1 a 3 -> Baixo | 4 a 6 -> Medio | 7 a 10 -> Alto
df_categorico = pd.DataFrame()

colunas_medicas = [c for c in df.columns if c != 'Class']
for col in colunas_medicas:
    df_categorico[col] = pd.cut(
        df[col], 
        bins=[0, 3, 6, 10], 
        labels=['Baixo', 'Medio', 'Alto']
    )

# Mantém a coluna de diagnóstico (benign / malignant)
df_categorico['Class'] = df['Class'].astype(str)

print("Amostra das variáveis categorizadas em intervalos (Baixo/Medio/Alto):")
print(df_categorico.head())

# ==============================================================================
# ETAPA 3: APLICAÇÃO DO ONE-HOT ENCODING
# ==============================================================================
print("\n--- ETAPA 3: Aplicação do One-Hot Encoding ---")

# Converte cada variável categórica em colunas binárias (0 ou 1)
df_encoded = pd.get_dummies(df_categorico, columns=df_categorico.columns)

print(f"Dimensão da matriz One-Hot Encoded: {df_encoded.shape}")
print("Amostra das colunas binárias geradas:")
print(list(df_encoded.columns[:8]))

# ==============================================================================
# ETAPA 4 E 5: GERAÇÃO E APRESENTAÇÃO DAS REGRAS COM MAIOR CONFIANÇA
# ==============================================================================
print("\n--- ETAPA 4 e 5: Geração de Regras de Associação (Apriori) ---")

if MLXTEND_DISPONIVEL:
    # Aplicação do algoritmo Apriori (Suporte mínimo = 15%)
    itemsets_frequentes = apriori(df_encoded, min_support=0.15, use_colnames=True)
    
    # Extração de regras com Confiança mínima de 80% (0.80)
    regras = association_rules(itemsets_frequentes, metric="confidence", min_threshold=0.80)
    
    # Ordenação das regras pela maior Confiança e pelo maior Lift
    regras_ordenadas = regras.sort_values(by=['confidence', 'lift'], ascending=[False, False])
    
    print(f"\nTotal de regras geradas: {len(regras_ordenadas)}")
    print("\nTop 10 Regras com Maior Confiança:")
    print("-" * 80)
    
    top_10 = regras_ordenadas.head(10)
    for idx, row in top_10.iterrows():
        ant = list(row['antecedents'])
        cons = list(row['consequents'])
        print(f"Regra: SE {ant} ENTÃO {cons}")
        print(f" -> Suporte: {row['support']:.2%} | Confiança: {row['confidence']:.2%} | Lift: {row['lift']:.2f}\n")

else:
    print("\n[Aviso] Instale a biblioteca 'mlxtend' para rodar a extração automática das regras.")