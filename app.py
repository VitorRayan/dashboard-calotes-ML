import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Calotes", layout="wide")

# ===== Estilo CSS moderno =====
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
        /* Cards KPIs */
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

# ===== Função para carregar dados diretamente do CSV =====
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv("UCI_Credit_Card.csv")
    df['Educação'] = df['EDUCATION'].map({1: 'Pós-graduação', 2: 'Universitário', 3: 'Ensino Médio',
                                          4: 'Outros', 5: 'Outros', 6: 'Outros', 0: 'Outros'})
    df['Sexo'] = df['SEX'].map({1: 'Homem', 2: 'Mulher'})
    df['Estado_Civil'] = df['MARRIAGE'].map({1: 'Casado(a)', 2: 'Solteiro(a)', 3: 'Outro'}).fillna('Outro')
    df['Calote'] = df['default.payment.next.month'].map({0: 0, 1: 1})  # para ML: 0 e 1
    df['Faixa_Idade'] = pd.cut(df['AGE'], bins=range(20, 81, 10), right=False)
    return df

# ===== Carregar dados =====
df = load_data()

# ===== Sidebar: Filtros =====
st.sidebar.header("🎛️ Filtros")

sexo = st.sidebar.multiselect("Sexo", df['Sexo'].unique(), default=df['Sexo'].unique())
educ = st.sidebar.multiselect("Educação", df['Educação'].unique(), default=df['Educação'].unique())
estado = st.sidebar.multiselect("Estado Civil", df['Estado_Civil'].unique(), default=df['Estado_Civil'].unique())

# Filtro faixa de idade slider
idade_min = int(df['AGE'].min())
idade_max = int(df['AGE'].max())
faixa_idade = st.sidebar.slider("Faixa de Idade", min_value=idade_min, max_value=idade_max, value=(idade_min, idade_max))

df_filtrado = df[
    (df['Sexo'].isin(sexo)) &
    (df['Educação'].isin(educ)) &
    (df['Estado_Civil'].isin(estado)) &
    (df['AGE'] >= faixa_idade[0]) & (df['AGE'] <= faixa_idade[1])
]

# ===== KPIs =====
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

# ===== Título =====
st.title("📊 Dashboard Interativo - Análise de Calotes")
st.markdown("Visualização interativa dos dados de crédito e inadimplência dos clientes.")

# ===== Gráficos =====

# 1. Calote por idade
fig1 = px.histogram(
    df_filtrado, x="AGE", color=df_filtrado['Calote'].map({0:'Não Deu Calote',1:'Deu Calote'}), nbins=20, barmode="stack",
    color_discrete_map={"Não Deu Calote": "green", "Deu Calote": "red"},
    labels={"AGE": "Idade"},
    title="Distribuição de Idade com Calote",
    hover_data={"AGE": True}
)
st.plotly_chart(fig1, use_container_width=True)

# 2. Limite por faixa de idade (convertendo intervalo para string)
df_filtrado['Faixa_Idade_Str'] = df_filtrado['Faixa_Idade'].astype(str)
limite = df_filtrado.groupby('Faixa_Idade_Str')['LIMIT_BAL'].mean().reset_index()
fig2 = px.bar(limite, x='Faixa_Idade_Str', y='LIMIT_BAL', title="Média de Limite por Faixa de Idade",
              labels={'LIMIT_BAL': 'Limite Médio', 'Faixa_Idade_Str': 'Faixa de Idade'},
              color='LIMIT_BAL', color_continuous_scale='Blues',
              hover_data={'LIMIT_BAL': ':.2f'})
st.plotly_chart(fig2, use_container_width=True)

# 3. Calote por Estado Civil
estado_data = df_filtrado.groupby(['Estado_Civil', 'Calote']).size().reset_index(name='Contagem')
estado_data['Calote'] = estado_data['Calote'].map({0:'Não Deu Calote',1:'Deu Calote'})
fig3 = px.bar(estado_data, x='Estado_Civil', y='Contagem', color='Calote', barmode='group',
              color_discrete_map={"Não Deu Calote": "green", "Deu Calote": "red"},
              title="Calote por Estado Civil",
              hover_data={'Contagem': True})
st.plotly_chart(fig3, use_container_width=True)

# 4. Calote por Educação
educ_data = df_filtrado.groupby(['Educação', 'Calote']).size().reset_index(name='Total')
educ_data['Calote'] = educ_data['Calote'].map({0:'Não Deu Calote',1:'Deu Calote'})
fig4 = px.bar(educ_data, x='Educação', y='Total', color='Calote', barmode='group',
              color_discrete_map={"Não Deu Calote": "green", "Deu Calote": "red"},
              title="Calote por Escolaridade",
              hover_data={'Total': True})
st.plotly_chart(fig4, use_container_width=True)

# ===== Seção de Machine Learning =====
st.header("🤖 Previsão de Calote com Random Forest")

# Preparar dados para ML
features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

X = df[features]
y = df['Calote']

# Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Treinar modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prever e mostrar métricas
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"**Acurácia do Modelo:** {acc:.2%}")

st.write("**Matriz de Confusão:**")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig_cm)

st.write("**Relatório de Classificação:**")
st.text(classification_report(y_test, y_pred))

# Importância das variáveis
importances = model.feature_importances_
feat_importances = pd.Series(importances, index=features)
fig_imp, ax = plt.subplots()
feat_importances.nlargest(10).plot(kind='barh', ax=ax)
ax.set_title('Top 10 Importância das Variáveis')
st.pyplot(fig_imp)

# ===== Previsão para novo cliente =====
st.subheader("Prever se um novo cliente dará calote")

with st.form("previsao_form"):
    st.write("Insira os dados do cliente:")

    LIMIT_BAL = st.number_input("Limite de crédito (LIMIT_BAL)", min_value=0, value=20000)
    SEX = st.selectbox("Sexo", options=[1, 2], format_func=lambda x: "Homem" if x == 1 else "Mulher")
    EDUCATION = st.selectbox("Educação", options=[0,1,2,3,4,5,6], format_func=lambda x: {
        0:'Outros',1:'Pós-graduação',2:'Universitário',3:'Ensino Médio',4:'Outros',5:'Outros',6:'Outros'
    }[x])
    MARRIAGE = st.selectbox("Estado Civil", options=[0,1,2,3], format_func=lambda x: {
        0:'Outro',1:'Casado(a)',2:'Solteiro(a)',3:'Outro'
    }[x])
    AGE = st.number_input("Idade", min_value=20, max_value=80, value=30)
    PAY_0 = st.number_input("PAY_0 (status pagamento mês 1)", min_value=-2, max_value=8, value=0)
    PAY_2 = st.number_input("PAY_2 (status pagamento mês 2)", min_value=-2, max_value=8, value=0)
    PAY_3 = st.number_input("PAY_3 (status pagamento mês 3)", min_value=-2, max_value=8, value=0)
    PAY_4 = st.number_input("PAY_4 (status pagamento mês 4)", min_value=-2, max_value=8, value=0)
    PAY_5 = st.number_input("PAY_5 (status pagamento mês 5)", min_value=-2, max_value=8, value=0)
    PAY_6 = st.number_input("PAY_6 (status pagamento mês 6)", min_value=-2, max_value=8, value=0)
    BILL_AMT1 = st.number_input("Valor fatura mês 1 (BILL_AMT1)", min_value=0, value=0)
    BILL_AMT2 = st.number_input("Valor fatura mês 2 (BILL_AMT2)", min_value=0, value=0)
    BILL_AMT3 = st.number_input("Valor fatura mês 3 (BILL_AMT3)", min_value=0, value=0)
    BILL_AMT4 = st.number_input("Valor fatura mês 4 (BILL_AMT4)", min_value=0, value=0)
    BILL_AMT5 = st.number_input("Valor fatura mês 5 (BILL_AMT5)", min_value=0, value=0)
    BILL_AMT6 = st.number_input("Valor fatura mês 6 (BILL_AMT6)", min_value=0, value=0)
    PAY_AMT1 = st.number_input("Pagamento mês 1 (PAY_AMT1)", min_value=0, value=0)
    PAY_AMT2 = st.number_input("Pagamento mês 2 (PAY_AMT2)", min_value=0, value=0)
    PAY_AMT3 = st.number_input("Pagamento mês 3 (PAY_AMT3)", min_value=0, value=0)
    PAY_AMT4 = st.number_input("Pagamento mês 4 (PAY_AMT4)", min_value=0, value=0)
    PAY_AMT5 = st.number_input("Pagamento mês 5 (PAY_AMT5)", min_value=0, value=0)
    PAY_AMT6 = st.number_input("Pagamento mês 6 (PAY_AMT6)", min_value=0, value=0)

    submit = st.form_submit_button("Prever")

if submit:
    novo_cliente = pd.DataFrame({
        'LIMIT_BAL': [LIMIT_BAL],
        'SEX': [SEX],
        'EDUCATION': [EDUCATION],
        'MARRIAGE': [MARRIAGE],
        'AGE': [AGE],
        'PAY_0': [PAY_0],
        'PAY_2': [PAY_2],
        'PAY_3': [PAY_3],
        'PAY_4': [PAY_4],
        'PAY_5': [PAY_5],
        'PAY_6': [PAY_6],
        'BILL_AMT1': [BILL_AMT1],
        'BILL_AMT2': [BILL_AMT2],
        'BILL_AMT3': [BILL_AMT3],
        'BILL_AMT4': [BILL_AMT4],
        'BILL_AMT5': [BILL_AMT5],
        'BILL_AMT6': [BILL_AMT6],
        'PAY_AMT1': [PAY_AMT1],
        'PAY_AMT2': [PAY_AMT2],
        'PAY_AMT3': [PAY_AMT3],
        'PAY_AMT4': [PAY_AMT4],
        'PAY_AMT5': [PAY_AMT5],
        'PAY_AMT6': [PAY_AMT6],
    })

    pred = model.predict(novo_cliente)[0]
    resultado = "Deu Calote" if pred == 1 else "Não Deu Calote"
    st.success(f"O modelo prevê que esse cliente: {resultado}")
