import streamlit as st
import requests
import datetime
import os
import plotly.graph_objects as go

API_URL = os.getenv("API_URL", "http://localhost:8000")

if "jwt_token" not in st.session_state:
    st.session_state["jwt_token"] = None


def autenticaion_window():
    """
    Renderiza a tela inicial de autenticação com as abas de Login e Registro de usuários.

    Gerencia a interface visual de captura de credenciais (e-mail e senha), realiza a 
    comunicação via requisição HTTP POST com a API para validação e, em caso de 
    sucesso, extrai e armazena o token JWT de acesso na sessão do Streamlit 
    (session_state), atualizando a página para liberar a área restrita.

    parametros:
    - Nenhum.

    retorno:
    - None: A função atua apenas na renderização da interface visual e mutação de estados do Streamlit.
    """
    st.title("Bem-vindo a aplicação de predição de evasão dos alunos da ONG Pede Passos Mágicos")

    aba_login, aba_regitro = st.tabs(["Login", "Registrar"])

    with aba_login:
        st.subheader("Acesse sua conta")
        email = st.text_input("E-mail")
        password = st.text_input("Senha", type="password")

        if st.button("Entrar"):
            data = {"username": email, "password": password}
            response_login = requests.post(
                f"{API_URL}/login",
                data=data
            )
            if response_login.status_code in [200, 201]:
                dados_da_api = response_login.json()
                token_real = dados_da_api.get("access_token")
                st.session_state["jwt_token"] = token_real
                st.success("Login efetuado! Carregando sistema...")
                st.rerun()
            else:
                st.error("Erro ao efetuar Login. Verifique as credencias e tente novamente.")

    with aba_regitro:
        st.subheader("Crie uma nova conta")
        name = st.text_input("Nome")
        last_name = st.text_input("Sobrenome")
        new_email = st.text_input("Novo E-mail")
        new_password = st.text_input("Nova senha", type="password")
        position = st.text_input("Cargo")
        today = datetime.date.today()
        date_of_birth = st.date_input(
            "Data de nascimento",
            min_value=datetime.date(1900, 1, 1),
            max_value=today,
            format="DD/MM/YYYY"
        )

        if st.button("Registrar"):
            response_register = requests.post(
                f"{API_URL}/register",
                json={
                    "name": name,
                    "last_name": last_name,
                    "email": new_email,
                    "password": new_password,
                    "date_of_birth": date_of_birth.isoformat(),
                    "position": position
                }
            )
            if response_register.status_code in [200, 201]:
                st.success("Usuário registrado com sucesso! Por favor, faça login na aba ao lado.")
            else:
                st.error("Erro ao registrar. Verifique as credencias e tente novamente.")

def admin_monitoring_window(headers):
    """
    Renderiza o painel interativo de monitoramento de Data Drift para administradores.

    Constrói a interface visual utilizando Streamlit, permitindo a seleção de uma 
    variável (feature) do modelo e uma janela de tempo (histórico em dias). Realiza 
    uma requisição HTTP GET autenticada para a rota /admin/monitoring/drift da API. 
    A partir da resposta JSON, exibe métricas estatísticas de avaliação (Score KS e P-Value), 
    alertas visuais dinâmicos caso ocorra deslocamento de dados, e renderiza um 
    histograma comparativo sobreposto utilizando a biblioteca Plotly para ilustrar a 
    diferença entre a base de treinamento e os dados da produção recente. Também gerencia 
    alertas de erro para acesso negado (403) ou falta de dados (404).

    parametros:
    - headers (dict): Dicionário contendo os cabeçalhos da requisição HTTP, essencialmente o token JWT ativo ("Authorization": "Bearer ...").

    retorno:
    - None: A função atua exclusivamente na construção da interface visual, plotagem de gráficos e execução de requisições web.
    """
    st.title("🛡️ Painel Administrativo: Monitoramento de Data Drift")
    st.write("Analise o deslocamento dos dados (Drift) entre a base de treinamento e a produção atual.")
    
    features_disponiveis = [
        "inde", "iaa", "ieg", "ips", "ida", "mat", "por", "ing", "ipv", "ian", 
        "idade", "diferenca_mat", "diferenca_por", "diferenca_ing", "diferenca_ieg", "diferenca_inde"
    ]
    col1, col2 = st.columns([3, 1])
    with col1:
        feature_select = st.selectbox(
            "Selecione a variável para analisar:",
            features_disponiveis
        )
    with col2:
        history_day = st.number_input(
            "Dias de histórico:",
            min_value=1,
            max_value=365,
            value=30
        )
        
    if st.button(f"Analisar Drift em '{feature_select}'"):
        with st.spinner("Extraindo dados do banco e calculando estatísticas..."):
            response = requests.get(
                f"{API_URL}/admin/monitoring/drift?feature={feature_select}&history_day={history_day}",
                headers=headers
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                if not data or not isinstance(data, dict):
                    st.error("A API retornou uma resposta vazia ou inválida.")
                    return
                c1, c2, c3 = st.columns(3)
                if data.get("drift_detected"):
                    c1.error("DRIFT DETECTADO")
                    st.warning("O comportamento dos alunos nesta variável mudou significativamente. O modelo pode precisar de retreinamento.")
                else:
                    c1.success("DADOS ESTÁVEIS")
                    
                c2.metric("Distância KS (Score)", data.get("drift_score_ks", "N/A"))
                c3.metric("P-Value", data.get("p_value", "N/A"))
                st.divider()
                
                data_graph = data.get("grafico_dados", {})
                if data_graph:
                    fig = go.Figure()
                    # Barras da Base de Treino (Azul translúcido)
                    fig.add_trace(go.Bar(
                        x=data_graph["eixo_x_categorias"],
                        y=data_graph["distribuicao_referencia"],
                        name='Treinamento (Baseline)',
                        marker_color='rgba(55, 128, 191, 0.6)'
                    ))
                    # Barras da Base de Produção (Laranja sólido)
                    fig.add_trace(go.Bar(
                        x=data_graph["eixo_x_categorias"],
                        y=data_graph["distribuicao_atual"],
                        name='Produção (Recente)',
                        marker_color='rgba(255, 153, 51, 0.9)'
                    ))
                    fig.update_layout(
                        title=f"Comparação de Distribuição: {feature_select.upper()}",
                        # Coloca as barras uma por cima da outra
                        barmode='overlay',
                        xaxis_title=f"Faixas de valor ({feature_select})",
                        yaxis_title="Quantidade de Alunos",
                        legend_title="Origem dos Dados"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            elif response.status_code == 403:
                st.error("Acesso Negado: Apenas o cargo 'admin' pode ver esta página.")
            elif response.status_code == 404:
                st.warning(f"Não há predições suficientes nos últimos {history_day} dias no banco de dados para analisar.")
            else:
                st.error(f"Erro na API: {response.status_code} - {response.text}")


def main_login_window():
    """
    Renderiza a interface principal da área restrita para usuários autenticados.

    Realiza a validação do token JWT ativo consultando a API (endpoint /me), exibe 
    os dados do usuário e gerencia a interface de upload de planilhas (Excel) para 
    predição. A função envia o arquivo para o modelo via requisição HTTP POST 
    e, em caso de sucesso, disponibiliza um botão para o download dos resultados 
    em formato CSV. Também gerencia a ação de encerramento de sessão (logout).

    parametros:
    - Nenhum.

    retorno:
    - None: A função atua exclusivamente na construção da interface visual, execução de requisições HTTP e gerenciamento de estado do Streamlit.
    """
    headers = {"Authorization": f"Bearer {st.session_state['jwt_token']}"}
    st.write("Cabeçalho que está sendo enviado:", headers)
    verify_autenticaion = requests.get(f"{API_URL}/me", headers=headers)

    if verify_autenticaion.status_code == 200:
        data_user = verify_autenticaion.json()
        position_user = data_user.get("position", "").lower()
        st.sidebar.title(f"Olá, {data_user.get('name')}!")
        st.sidebar.write(f"Cargo: **{position_user.title()}**")
        st.sidebar.divider()
        menu_option = ["Realizar Predições"]
        if position_user == "admin":
            menu_option.append("Painel de Monitoramento (Drift)")
            
        select_menu = st.sidebar.radio("Navegação", menu_option)
        
        st.sidebar.divider()
        if st.sidebar.button("Sair (Logout)"):
            st.session_state["jwt_token"] = None
            st.rerun()
        if select_menu == "Painel de Monitoramento (Drift)":
            admin_monitoring_window(headers)

        elif select_menu == "Realizar Predições":
            st.title("Área Restrita a usuários logados: Previsão de evasão")
            st.divider()

            file = st.file_uploader("Envie os dados para predição (Planilha de dados dos alunos)", type=["xlsx", "xls"])

            if file and st.button("Gerar predição"):
                st.info("Enviando para a API com autenticação JWT...")
                with st.spinner("Enviando dados para o modelo. Isso pode levar alguns segundos, aguarde..."):
                    files = {
                        "file": (
                            file.name,
                            file.getvalue(),
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    }
                    response_prediction = requests.post(
                        f"{API_URL}/predict/batch",
                        headers=headers,
                        files=files
                    )

                    if response_prediction.status_code in [200, 201]:
                        st.success("Processamento concluído com sucesso!")
                        file_name_ori = file.name
                        file_name_csv = file_name_ori.rsplit(".", 1)[0] + "_previsoes.csv"

                        st.download_button(
                            label="📥 Descarregar Folha de Cálculo com Previsões",
                            data=response_prediction.content, 
                            file_name=file_name_csv,
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Ocorreu um erro na API (Código {response_prediction.status_code})")
                        st.write(response_prediction.text)
    else:
        st.error("Sua sessão expirou ou é inválida. Por favor, faça login novamente.")
        st.session_state["jwt_token"] = None
        if st.button("Voltar ao Login"):
            st.rerun()


if st.session_state["jwt_token"] is None:
    autenticaion_window()
else:
    main_login_window()
