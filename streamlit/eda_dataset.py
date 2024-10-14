import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações iniciais
st.set_page_config(layout="wide")
sns.set_style('whitegrid')

# Função para carregar os dados
@st.cache
def load_data():
    url = '../env/dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    data = pd.read_csv(url)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['SeniorCitizen'] = data['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})
    data.dropna(inplace=True)
    return data

# Função principal do Streamlit
def main():
    # Título e Introdução
    st.title("Análise Exploratória: Telco Customer Churn")
    st.markdown("""
    **Contexto:** Esta análise explora dados de clientes de uma empresa de telecomunicações,
    buscando entender os fatores que influenciam o cancelamento de serviços (*churn*).
    Vamos investigar como diferentes características dos clientes estão relacionadas ao churn.
    """)

    # Carregar os dados
    data = load_data()

    # Visão Geral dos Dados
    st.header("Visão Geral dos Dados")
    st.write("A base de dados contém informações de {} clientes.".format(data.shape[0]))
    st.write(data.head())

    # Distribuição da Variável Alvo (Churn)
    st.subheader("Distribuição do Churn")
    fig1, ax1 = plt.subplots()
    churn_counts = data['Churn'].value_counts(normalize=True) * 100
    sns.barplot(x=churn_counts.index, y=churn_counts.values, palette='viridis', ax=ax1)
    ax1.set_ylabel('Percentual')
    ax1.set_title('Percentual de Clientes que Realizaram Churn')
    for i, v in enumerate(churn_counts.values):
        ax1.text(i, v + 1, f"{v:.2f}%", ha='center')
    st.pyplot(fig1)

    # Análise Demográfica
    st.header("Análise Demográfica")

    # Configuração de layout para múltiplos gráficos
    col1, col2, col3 = st.columns(3)

    # Gênero
    with col1:
        st.subheader("Gênero")
        gender_churn = data.groupby('gender')['Churn'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
        fig2, ax2 = plt.subplots()
        sns.barplot(x='gender', y='percent', hue='Churn', data=gender_churn, palette='pastel', ax=ax2)
        ax2.set_title('Churn por Gênero (%)')
        st.pyplot(fig2)

    # Senior Citizen
    with col2:
        st.subheader("Idosos (Senior Citizens)")
        senior_churn = data.groupby('SeniorCitizen')['Churn'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
        fig3, ax3 = plt.subplots()
        sns.barplot(x='SeniorCitizen', y='percent', hue='Churn', data=senior_churn, palette='Set2', ax=ax3)
        ax3.set_title('Churn por Idosos (%)')
        st.pyplot(fig3)

    # Dependentes
    with col3:
        st.subheader("Dependentes")
        dependents_churn = data.groupby('Dependents')['Churn'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
        fig4, ax4 = plt.subplots()
        sns.barplot(x='Dependents', y='percent', hue='Churn', data=dependents_churn, palette='coolwarm', ax=ax4)
        ax4.set_title('Churn por Dependentes (%)')
        st.pyplot(fig4)

    st.markdown("""
    **Observações:**
    - **Gênero:** Não há diferença significativa na taxa de churn entre homens e mulheres.
    - **Idosos:** Clientes idosos têm uma taxa de churn mais alta.
    - **Dependentes:** Clientes sem dependentes tendem a cancelar mais os serviços.
    """)

    # Análise dos Serviços
    st.header("Análise dos Serviços")

    col4, col5, col6 = st.columns(3)

    # Internet Service
    with col4:
        st.subheader("Tipo de Internet")
        internet_churn = data.groupby('InternetService')['Churn'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
        fig5, ax5 = plt.subplots()
        sns.barplot(x='InternetService', y='percent', hue='Churn', data=internet_churn, palette='Accent', ax=ax5)
        ax5.set_title('Churn por Tipo de Internet (%)')
        plt.xticks(rotation=45)
        st.pyplot(fig5)

    # Contract Type
    with col5:
        st.subheader("Tipo de Contrato")
        contract_churn = data.groupby('Contract')['Churn'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
        fig6, ax6 = plt.subplots()
        sns.barplot(x='Contract', y='percent', hue='Churn', data=contract_churn, palette='Dark2', ax=ax6)
        ax6.set_title('Churn por Tipo de Contrato (%)')
        plt.xticks(rotation=45)
        st.pyplot(fig6)

    # Payment Method
    with col6:
        st.subheader("Método de Pagamento")
        payment_churn = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
        fig7, ax7 = plt.subplots()
        sns.barplot(x='PaymentMethod', y='percent', hue='Churn', data=payment_churn, palette='Set1', ax=ax7)
        ax7.set_title('Churn por Método de Pagamento (%)')
        plt.xticks(rotation=45)
        st.pyplot(fig7)

    st.markdown("""
    **Observações:**
    - **Tipo de Internet:** Clientes com fibra ótica apresentam maior churn.
    - **Tipo de Contrato:** Contratos mensais têm taxas de churn significativamente maiores.
    - **Método de Pagamento:** Clientes que pagam com débito automático têm menor churn.
    """)

    # Análise Financeira
    st.header("Análise Financeira")

    # Distribuição dos Charges
    fig8, ax8 = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data[data['Churn'] == 'No']['MonthlyCharges'], label='Não Churn', shade=True)
    sns.kdeplot(data[data['Churn'] == 'Yes']['MonthlyCharges'], label='Churn', shade=True)
    ax8.set_title('Distribuição de Cobranças Mensais')
    ax8.set_xlabel('Cobrança Mensal')
    ax8.legend()
    st.pyplot(fig8)

    st.markdown("""
    **Observações:**
    - Clientes que realizam churn tendem a ter cobranças mensais mais altas.
    """)

    # Análise do Tempo de Permanência (Tenure)
    st.header("Tempo de Permanência e Churn")

    fig9, ax9 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x='tenure', hue='Churn', multiple='stack', bins=30, palette='muted', ax=ax9)
    ax9.set_title('Distribuição do Tempo de Permanência')
    ax9.set_xlabel('Meses com a Empresa')
    st.pyplot(fig9)

    st.markdown("""
    **Observações:**
    - Clientes com menor tempo de permanência têm maior probabilidade de churn.
    - Estratégias de retenção devem focar nos clientes novos.
    """)

    # Análise de Métodos de Pagamento
    st.markdown("---")
    st.header("Análise de Métodos de Pagamento")

    col1, col2 = st.columns(2)

    with col1:
        # Distribuição dos Métodos de Pagamento
        st.subheader("Distribuição dos Métodos de Pagamento")
        payment_counts = data['PaymentMethod'].value_counts()
        plt.figure(figsize=(8,6))
        ax = sns.countplot(y='PaymentMethod', data=data, palette='Set2', order=payment_counts.index)
        plt.title('Contagem dos Métodos de Pagamento')
        plt.xlabel('Contagem')
        plt.ylabel('Método de Pagamento')

        # Adicionar labels nas barras
        for p in ax.patches:
            width = p.get_width()
            plt.text(width + 1, p.get_y() + p.get_height()/2, int(width), va='center')

        st.pyplot(plt)
        st.write(f"Distribuição dos Métodos de Pagamento:\n{payment_counts}")

    with col2:
        # Churn por Método de Pagamento
        st.subheader("Churn por Método de Pagamento")
        payment_churn = data.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()
        ax = payment_churn.plot(kind='barh', stacked=True, figsize=(8,6), color=['green', 'red'])
        plt.title('Proporção de Churn por Método de Pagamento')
        plt.xlabel('Proporção')
        plt.ylabel('Método de Pagamento')
        plt.legend(title='Churn', loc='lower right')

        # Adicionar labels nas barras
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='center')

        st.pyplot(plt)
        st.write("""
        Observamos que clientes que utilizam **Pagamento Eletrônico** (Electronic Check) têm uma taxa de churn significativamente maior. Métodos automatizados, como débito em conta bancária e cartão de crédito, apresentam taxas de churn menores.
        """)


    # Análise de Faturamento Sem Papel (PaperlessBilling)
    st.subheader("Churn por Faturamento Sem Papel")
    paperless_churn = data.groupby('PaperlessBilling')['Churn'].value_counts(normalize=True).unstack()
    paperless_churn.plot(kind='bar', stacked=True, figsize=(6,4), color=['green', 'red'])
    plt.title('Proporção de Churn por Faturamento Sem Papel')
    plt.xlabel('Faturamento Sem Papel')
    plt.ylabel('Proporção')
    plt.legend(title='Churn', loc='upper right')
    st.pyplot(plt)
    st.write("""
    Clientes que optam pelo **Faturamento Sem Papel** têm uma taxa de churn mais alta. Isso pode estar relacionado ao perfil desses clientes ou à forma como recebem e entendem suas cobranças.
    """)

    # Análise de Serviços Adicionais
    st.markdown("---")
    st.header("Análise de Serviços Adicionais")

    # Listar serviços adicionais
    additional_services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

    for service in additional_services:
        st.subheader(f"Churn por {service}")
        service_churn = data.groupby(service)['Churn'].value_counts(normalize=True).unstack()
        service_churn.plot(kind='bar', stacked=True, figsize=(6,4), color=['green', 'red'])
        plt.title(f'Proporção de Churn por {service}')
        plt.xlabel(service)
        plt.ylabel('Proporção')
        plt.legend(title='Churn', loc='upper right')
        st.pyplot(plt)
        st.write(f"""
        Analisando o serviço **{service}**, podemos observar:
        """)
        # Comentários específicos para cada serviço (opcional)
        if service == 'OnlineSecurity':
            st.write("""
            Clientes sem **Segurança Online** têm uma taxa de churn maior. Oferecer esse serviço pode aumentar a retenção.
            """)
        elif service == 'TechSupport':
            st.write("""
            A falta de **Suporte Técnico** está associada a uma taxa de churn mais alta, indicando a importância de suporte ao cliente.
            """)
        # Adicionar mais comentários conforme necessário

    # Análise de Parceiros (Partner)
    st.markdown("---")
    st.header("Análise de Parceria")

    st.subheader("Churn por Parceiros")
    partner_churn = data.groupby('Partner')['Churn'].value_counts(normalize=True).unstack()
    partner_churn.plot(kind='bar', stacked=True, figsize=(6,4), color=['green', 'red'])
    plt.title('Proporção de Churn por Parceiros')
    plt.xlabel('Possui Parceiro')
    plt.ylabel('Proporção')
    plt.legend(title='Churn', loc='upper right')
    st.pyplot(plt)
    st.write("""
    Clientes que **não possuem parceiro** têm uma taxa de churn maior. Isso pode indicar que pessoas em relacionamentos comprometidos são mais estáveis em suas escolhas de serviços.
    """)



    # Correlação entre Variáveis Numéricas
    st.header("Correlação entre Variáveis Numéricas")

    num_vars = ['tenure', 'MonthlyCharges', 'TotalCharges']
    corr_data = data[num_vars]
    corr_data['TotalCharges'] = pd.to_numeric(corr_data['TotalCharges'], errors='coerce')
    corr = corr_data.corr()

    fig10, ax10 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='Blues', ax=ax10)
    ax10.set_title('Matriz de Correlação')
    st.pyplot(fig10)

    st.markdown("""
    **Observações:**
    - Existe uma correlação positiva entre `tenure` e `TotalCharges`, o que é esperado.
    - `MonthlyCharges` tem correlação baixa com `tenure`, indicando que clientes novos podem ter cobranças similares aos antigos.
    """)

    # Conclusões Gerais
    st.header("Conclusões Gerais")
    st.markdown("""
    A análise exploratória revelou insights importantes sobre os fatores que influenciam o churn:

    - **Perfil Demográfico:** Idosos e clientes sem dependentes apresentam maior churn.
    - **Serviços Contratados:** Clientes com fibra ótica e contratos mensais têm maior propensão ao churn.
    - **Aspectos Financeiros:** Cobranças mensais mais altas estão associadas ao churn.
    - **Tempo de Permanência:** Clientes recentes estão mais propensos a cancelar.

    **Recomendações:**

    - Desenvolver programas de fidelidade para novos clientes.
    - Oferecer incentivos para clientes com contratos mensais migrarem para contratos de longo prazo.
    - Avaliar ofertas especiais para clientes com cobranças mensais altas.
    """)

if __name__ == '__main__':
    main()
