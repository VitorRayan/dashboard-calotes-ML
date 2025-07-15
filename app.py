import os
import pandas as pd
import streamlit as st
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Dashboard Calotes com ML", layout="wide")

# ===== Estilo CSS =====
st.markdown("""
    <style>
        .element-container {
            transition: transform 0.3s ease;
        }
        .element-container:hover {
            transform: scale(1.02);
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }
        .block-container {
            padding-top: 2rem;
        }
        .stButton>button {
            color: white;
            background: #0083B8;
            border-radius: 8px;
        }
        .kpi {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: bold;
            color: #0083B8;
        }
        .kpi-label {
            font-size: 1rem;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=True)
def load_data():
    dataset = 'uciml/default-of-credit-card-clients-dataset'
    file_name = 'UCI_Credit_Card.csv'
    zip_file = 'default-of-credit-card-clients-dataset.zip'

    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(file_name):
        if not os.path.exists(zip_file):
            api.dataset_download_files(dataset, path='.', unzip=False)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall('.')

    df = pd.read_csv(file_name)
    df['Educação'] = df['EDUCATION'].map({1: 'Pós-graduação', 2: 'Universitário', 3: 'Ensino Médio',
                                          4: 'Outros', 5: 'Outros', 6: 'Outros', 0: 'Outros'})
    df['Sexo'] = df['SEX'].map({1: 'Homem', 2: 'Mulher'})
    df['Estado_Civil'] = df['MARRIAGE'].map({1: 'Casado(a)', 2: 'Solteiro(a)', 3: 'Outro'}).fillna('Outro')
    df['Calote'] = df['default.payment.next.month'].map({0: 0, 1: 1})  # para ML: 0 e 1
    df['Faixa_Idade'] = pd.cut(df['AGE'], bins=range(20, 81, 10), right=False)
    return df

df = load_data()

# Sidebar filtros
st.sidebar.header("🎛️ Filtros")
sexo = st.sidebar.multiselect("Sexo", df['Sexo'].unique(), default=df['Sexo'].unique())
educ = st.sidebar.multiselect("Educação", df['Educação'].unique(), default=df['Educação'].unique())
estado = st.sidebar.multiselect("Estado Civil", df['Estado_Civil'].unique(), default=df['Estado_Civil'].unique())
idade_min = int(df['AGE'].min())
idade_max = int(df['AGE'].max())
faixa_idade = st.sidebar.slider("Faixa de Idade", min_value=idade_min, max_value=idade_max, value=(idade_min, idade_max))

df_filtrado = df[
    (df['Sexo'].isin(sexo)) &
    (df['Educação'].isin(educ)) &
    (df['Estado_Civil'].isin(estado)) &
    (df['AGE'] >= faixa_idade[0]) & (df['AGE'] <= faixa_idade[1])
]

# KPIs
total_clientes = len(df_filtrado)
pct_calote = 0 if total_clientes == 0 else (df_filtrado['Calote'] == 1).mean() * 100
media_limite = 0 if total_clientes == 0 else df_filtrado['LIMIT_BAL'].mean()

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.markdown(f"""
    <div class="kpi">
        <div class="kpi-value">{total_clientes}</div>
        <div class="kpi-label">Clientes Filtrados</div>
    </div>
""", unsafe_allow_html=True)
kpi2.markdown(f"""
    <div class="kpi">
        <div class="kpi-value">{pct_calote:.2f}%</div>
        <div class="kpi-label">Percentual de Calote</div>
    </div>
""", unsafe_allow_html=True)
kpi3.markdown(f"""
    <div class="kpi">
        <div class="kpi-value">R$ {media_limite:,.2f}</div>
        <div class="kpi-label">Limite Médio</div>
    </div>
""", unsafe_allow_html=True)

# Título
st.title("📊 Dashboard Interativo - Análise de Calotes com Machine Learning")
st.markdown("Visualização interativa dos dados e previsão de calote para novos clientes.")

# Gráficos
fig1 = px.histogram(
    df_filtrado, x="AGE", color=df_filtrado['Calote'].map({0:'Não Deu Calote',1:'Deu Calote'}), nbins=20, barmode="stack",
    color_discrete_map={"Não Deu Calote": "green", "Deu Calote": "red"},
    labels={"AGE": "Idade"},
    title="Distribuição de Idade com Calote",
    hover_data={"AGE": True, "Calote": True}
)
st.plotly_chart(fig1, use_container_width=True)

df_filtrado['Faixa_Idade_Str'] = df_filtrado['Faixa_Idade'].astype(str)
limite = df_filtrado.groupby('Faixa_Idade_Str')['LIMIT_BAL'].mean().reset_index()
fig2 = px.bar(limite, x='Faixa_Idade_Str', y='LIMIT_BAL', title="Média de Limite por Faixa de Idade",
              labels={'LIMIT_BAL': 'Limite Médio', 'Faixa_Idade_Str': 'Faixa de Idade'},
              color='LIMIT_BAL', color_continuous_scale='Blues',
              hover_data={'LIMIT_BAL': ':.2f'})
st.plotly_chart(fig2, use_container_width=True)

estado_data = df_filtrado.groupby(['Estado_Civil', 'Calote']).size().reset_index(name='Contagem')
estado_data['Calote_Label'] = estado_data['Calote'].map({0:'Não Deu Calote',1:'Deu Calote'})
fig3 = px.bar(estado_data, x='Estado_Civil', y='Contagem', color='Calote_Label', barmode='group',
              color_discrete_map={"Não Deu Calote": "green", "Deu Calote": "red"},
              title="Calote por Estado Civil",
              hover_data={'Contagem': True})
st.plotly_chart(fig3, use_container_width=True)

educ_data = df_filtrado.groupby(['Educação', 'Calote']).size().reset_index(name='Total')
educ_data['Calote_Label'] = educ_data['Calote'].map({0:'Não Deu Calote',1:'Deu Calote'})
fig4 = px.bar(educ_data, x='Educação', y='Total', color='Calote_Label', barmode='group',
              color_discrete_map={"Não Deu Calote": "green", "Deu Calote": "red"},
              title="Calote por Escolaridade",
              hover_data={'Total': True})
st.plotly_chart(fig4, use_container_width=True)

# Tabela final
st.subheader("📋 Tabela com Dados Filtrados")
st.dataframe(df_filtrado[['ID', 'Sexo', 'Educação', 'Estado_Civil', 'AGE', 'LIMIT_BAL', 'Calote']].reset_index(drop=True))


# ======= Machine Learning =======
st.header("⚙️ Treinamento e Previsão com Random Forest")

# Preparar dados para ML
features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# Usar valores originais numéricos para ML
X = df[features]
y = df['Calote']

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Botão para treinar o modelo
if st.button("Treinar Modelo"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Acurácia no conjunto de teste:** {acc*100:.2f}%")

    st.write("**Matriz de Confusão:**")
    st.text(confusion_matrix(y_test, y_pred))

    st.write("**Relatório de Classificação:**")
    st.text(classification_report(y_test, y_pred))

    importances = rf.feature_importances_
    feat_importances = pd.Series(importances, index=features).sort_values(ascending=False)
    st.write("**Importância das Variáveis:**")
    st.bar_chart(feat_importances)

    # Salvar modelo no estado para usar na previsão
    st.session_state['rf_model'] = rf

# Interface para previsão de novo cliente
st.header("🔮 Prever Calote para Novo Cliente")

if 'rf_model' not in st.session_state:
    st.warning("Treine o modelo antes de fazer previsões.")
else:
    modelo = st.session_state['rf_model']

    with st.form("form_prever"):
        st.write("Informe os dados do novo cliente:")

        limit_bal = st.number_input("Limite de Crédito (LIMIT_BAL)", min_value=0, value=50000, step=1000)
        sex = st.selectbox("Sexo (SEX)", options=[1, 2], format_func=lambda x: "Homem" if x==1 else "Mulher")
        education = st.selectbox("Educação (EDUCATION)", options=[0,1,2,3,4,5,6],
                                 format_func=lambda x: {0:'Outros',1:'Pós-graduação',2:'Universitário',3:'Ensino Médio',4:'Outros',5:'Outros',6:'Outros'}.get(x))
        marriage = st.selectbox("Estado Civil (MARRIAGE)", options=[0,1,2,3], format_func=lambda x: {0:'Outro',1:'Casado(a)',2:'Solteiro(a)',3:'Outro'}.get(x))
        age = st.number_input("Idade (AGE)", min_value=21, max_value=79, value=30)

        pay_0 = st.slider("Pagamento Mês 0 (PAY_0)", min_value=-2, max_value=8, value=0)
        pay_2 = st.slider("Pagamento Mês 2 (PAY_2)", min_value=-2, max_value=8, value=0)
        pay_3 = st.slider("Pagamento Mês 3 (PAY_3)", min_value=-2, max_value=8, value=0)
        pay_4 = st.slider("Pagamento Mês 4 (PAY_4)", min_value=-2, max_value=8, value=0)
        pay_5 = st.slider("Pagamento Mês 5 (PAY_5)", min_value=-2, max_value=8, value=0)
        pay_6 = st.slider("Pagamento Mês 6 (PAY_6)", min_value=-2, max_value=8, value=0)

        bill_amt1 = st.number_input("Valor Fatura Mês 1 (BILL_AMT1)", min_value=0, value=0, step=100)
        bill_amt2 = st.number_input("Valor Fatura Mês 2 (BILL_AMT2)", min_value=0, value=0, step=100)
        bill_amt3 = st.number_input("Valor Fatura Mês 3 (BILL_AMT3)", min_value=0, value=0, step=100)
        bill_amt4 = st.number_input("Valor Fatura Mês 4 (BILL_AMT4)", min_value=0, value=0, step=100)
        bill_amt5 = st.number_input("Valor Fatura Mês 5 (BILL_AMT5)", min_value=0, value=0, step=100)
        bill_amt6 = st.number_input("Valor Fatura Mês 6 (BILL_AMT6)", min_value=0, value=0, step=100)

        pay_amt1 = st.number_input("Valor Pagamento Mês 1 (PAY_AMT1)", min_value=0, value=0, step=100)
        pay_amt2 = st.number_input("Valor Pagamento Mês 2 (PAY_AMT2)", min_value=0, value=0, step=100)
        pay_amt3 = st.number_input("Valor Pagamento Mês 3 (PAY_AMT3)", min_value=0, value=0, step=100)
        pay_amt4 = st.number_input("Valor Pagamento Mês 4 (PAY_AMT4)", min_value=0, value=0, step=100)
        pay_amt5 = st.number_input("Valor Pagamento Mês 5 (PAY_AMT5)", min_value=0, value=0, step=100)
        pay_amt6 = st.number_input("Valor Pagamento Mês 6 (PAY_AMT6)", min_value=0, value=0, step=100)

        enviar = st.form_submit_button("Prever Calote")

    if enviar:
        entrada = pd.DataFrame({
            'LIMIT_BAL': [limit_bal],
            'SEX': [sex],
            'EDUCATION': [education],
            'MARRIAGE': [marriage],
            'AGE': [age],
            'PAY_0': [pay_0],
            'PAY_2': [pay_2],
            'PAY_3': [pay_3],
            'PAY_4': [pay_4],
            'PAY_5': [pay_5],
            'PAY_6': [pay_6],
            'BILL_AMT1': [bill_amt1],
            'BILL_AMT2': [bill_amt2],
            'BILL_AMT3': [bill_amt3],
            'BILL_AMT4': [bill_amt4],
            'BILL_AMT5': [bill_amt5],
            'BILL_AMT6': [bill_amt6],
            'PAY_AMT1': [pay_amt1],
            'PAY_AMT2': [pay_amt2],
            'PAY_AMT3': [pay_amt3],
            'PAY_AMT4': [pay_amt4],
            'PAY_AMT5': [pay_amt5],
            'PAY_AMT6': [pay_amt6],
        })

        pred = modelo.predict(entrada)[0]

        if pred == 0:
            st.success("✅ O modelo prevê que o cliente NÃO dará calote.")
        else:
            st.error("⚠️ O modelo prevê que o cliente DARÁ calote.")