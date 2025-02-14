#-------------------------------------------------------
#  Aplicação Streamlit
#  Disponibiliza o Modelo de Clusterização construido
#-------------------------------------------------------

# Imports

import streamlit as st
import pandas as pd
import joblib

# Carga dos arquivos criados pelo Projeto_marketing.ipynb

encoder = joblib.load('encoder.pkl')       # OneHotEncoder
scaler = joblib.load('scaler.pkl')         # Normalização
kmeans = joblib.load('kmeans.pkl')         # Modelo

# Função para processar o arquivo recebido

def processar_prever(df):

    # Aplicar o encoder da coluna sexo
    encoded_sexo = encoder.transform(df[['sexo']])
    
    # Aplicar a normalização com o scaler
    encoded_df = pd.DataFrame(encoded_sexo, columns=encoder.get_feature_names_out(['sexo']))

    # Fazer a previsão dos dados com o modelo k-means
    dados = pd.concat([df.drop('sexo', axis=1), encoded_df], axis=1)

    # Aplicar a normalização
    dados_escalados = scaler.transform(dados)

    # Fazer as previsões
    cluster = kmeans.predict(dados_escalados)

    return cluster

# Titulo da Página, Explicação, Coleta do Arquivo, Processamento do Model e Devolução dos Resultados

st.title('Grupos de interesse para marketing')
st.write("""
         Neste projeto, aplicamos o algoritmo de clusterização K-means para identificar e prever agrupamentos de interesses de usuários, com o objetivo de direcionar campanhas de marketing de forma mais eficaz.
         Através dessa análise, conseguimos segmentar o público em bolhas de interesse, permitindo a criação de campanhas personalizadas e mais assertivas, com base nos padrões de comportamento e preferências de cada grupo.
         """)

up_file = st.file_uploader('Escolha um arquivo CSV para realizar a previsão',type='csv')


# Ocorre quando algm arquivo é enviado

if up_file is not None:

    st.write("""
                    ### Descrição dos Grupos:
                    - **Grupo 0** : É focado em um público jovem com forte interesse em moda, música e aparência.
                    - **Grupo 1** : Está muito associado a esportes, especialmente futebol americano, basquete e atividades culturais como banda e rock.
                    - **Grupo 2** : É mais equilibrado, com interesses em música, dança, e moda.
                """)

    # Cria um dataframe
    df = pd.read_csv(up_file)

    # Chamada para a execução do modelo
    cluster = processar_prever(df)

    # Insere uma coluna cluster no dataframe
    df.insert(0, 'grupos', cluster)
    
    st.write('Visualização dos resultados (10 primeiros registros):')
    st.write(df.head(10))
    
    csv = df.to_csv(index=False)
    st.download_button(label='Baixar resultados completos', data = csv, file_name = 'Grupos_Interesse.csv', mime='text/csv')
