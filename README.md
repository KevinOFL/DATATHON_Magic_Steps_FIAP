# 📘 FIAP DATATHON ML Engineer - Pede Passos Mágicos

Este repositório corresponde à entrega final da quinta e ultima fase do curso de pós-graduação em Engenharia de Machine Learning. Ele tem como objetivo apresentar uma solução com um modelo de predição treinado e possivelmente colocado em produção, aplicando todos os conceitos e habilidades desenvolvidas durante este curso.

---

## 📌 Sobre o Projeto

A solução é composta por um pipeline preditivo em desenvolvimento, destacando as seguintes entregas e tecnologias:

- **Treinamento e Registro de Modelos**: Foram treinados modelos de classificação robustos, como **XGBoost** e **Random Forest**, utilizando `RandomizedSearchCV` para busca de hiperparâmetros e tratamento de classes desbalanceadas. O **MLflow** foi utilizado como espinha dorsal para o rastreamento de experimentos, salvamento de artefatos (matrizes de confusão e feature importance) e versionamento de modelos ajustados com réguas de corte de probabilidade.
- **API REST com FastAPI e Segurança**: Construção de um backend de alta performance responsável por servir o modelo e orquestrar as predições em lote via processamento de arquivos Excel. A API possui um sistema completo de segurança com autenticação JWT e controle de acesso baseado em cargos (RBAC).
- **Banco de Dados (SQLAlchemy)**: Todo o histórico de requisições, cadastros de usuários e as predições individuais de cada aluno são persistidos em um banco de dados relacional (PostgreSQL/SQLite) utilizando operações de *bulk insert* para garantir a velocidade da API.
- **Frontend Interativo com Streamlit**: Uma interface amigável desenvolvida para que os analistas da ONG possam fazer o login, enviar as planilhas de dados brutos e efetuar o download das previsões formatadas de forma simples.
- **Monitoramento Contínuo (Data Drift)**: Implementação de um painel administrativo exclusivo no front-end, conectado a uma rota analítica da API. O sistema extrai as predições recentes do banco de dados e calcula o descolamento estatístico (Data Drift) em relação à base de treinamento utilizando o teste de Kolmogorov-Smirnov (KS) e gráficos interativos do Plotly, alertando sobre a necessidade de retreinamento do modelo.

## ⚙️ Tecnologias Utilizadas

| Tecnologia | Função |
| :--- | :--- |
| **FastAPI / Uvicorn** | Framework de alta performance e servidor ASGI para construção e execução da API REST. |
| **Streamlit** | Framework utilizado para o desenvolvimento do frontend interativo e painel de monitoramento. |
| **PostgreSQL / SQLAlchemy** | Banco de dados relacional e ORM para persistência de usuários, predições e histórico. |
| **XGBoost** | Algoritmo principal de *Gradient Boosting* treinado para a classificação do risco de evasão. |
| **Scikit-Learn** | Ferramentas de pré-processamento, métricas e treinamento de modelos baseline (*Random Forest*). |
| **MLflow** | Plataforma MLOps para gerenciar o ciclo de vida dos modelos (rastreamento, versionamento e artefatos). |
| **Pandas / NumPy** | Manipulação, limpeza e análise matemática estruturada dos dados tabulares da ONG. |
| **Plotly** | Criação de gráficos interativos (utilizado ativamente na renderização do painel de *Data Drift*). |
| **Matplotlib / Seaborn** | Geração de visualizações estáticas de avaliação, como matrizes de confusão e *feature importance*. |
| **Pytest** | Framework utilizado para a criação e execução automatizada de testes unitários. |
| **Python-JOSE / Pydantic** | Validação estrutural de dados, criptografia de senhas e gerenciamento de autenticação via JWT. |
| **OpenPyXL** | Biblioteca essencial para a leitura e manipulação das planilhas Excel enviadas pelos analistas. |
| **Black / Isort / Taskipy** | Ferramentas de desenvolvimento para formatação de código, *linting* e automação de tarefas (*scripts*). |

## 🚀 Como Executar Localmente

### 📦 Pré-requisitos

Certifique-se de ter as seguintes ferramentas instaladas no seu sistema:

* [Python 3.13+](https://www.python.org/downloads/)
* [Poetry](https://python-poetry.org/docs/) (Para gerenciamento de dependências virtuais)
* [Docker](https://www.docker.com/products/docker-desktop/) (Para rodar o contêiner do banco de dados)

### 🔧 Passo a Passo (Setup)

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/KevinOFL/DATATHON_Magic_Steps_FIAP.git
   cd SEU_REPOSITORIO
   ```
2. **Construa os conteiners:**
   ```bash
   docker-compose up -d
   ```
3. **Acessando o banco PostgreSQL de dados pelo cmd (Fora do conteiner Docker):**
   ```bash
   docker exec -it postgres-db psql -U admin -d datathon_db
   ```
4. **Rota de acesso ao MLFlow:**
    http://localhost:5000/
5. **Rota de acesso ao FASTApi Docs:**
    http://localhost:8000/docs
6. **Rota de acesso a aplicação StreamLit:**
    http://localhost:8501/

## 📌 Boas Práticas Aplicadas e Arquitetura MLOps

- **Engenharia de Features Robusta**: Criação de variáveis analíticas de discrepância de notas e indicadores (INDE, IEG) e tratamento dinâmico de nulos (imputação por mediana) para maximizar o poder preditivo.
- **Modularização do Pipeline**: Funções de pré-processamento estritamente reutilizáveis (`loading_data`), garantindo que os dados de inferência passem pelas exatas mesmas transformações dos dados de treino.
- **Tratamento de Desbalanceamento**: Uso de pesos de classe (`scale_pos_weight` e `class_weight="balanced"`) nos modelos Random Forest e XGBoost para lidar com a assimetria natural de alunos evadidos vs. retidos.
- **Métricas Focadas no Negócio**: Avaliação de modelos utilizando "Réguas de Corte" (Thresholds) customizadas, otimizando o *Recall* para garantir que o máximo de alunos em risco sejam detectados e salvos.
- **API REST Segura (FastAPI)**: Endpoints protegidos por autenticação JWT e senhas criptografadas (Python-JOSE/Passlib).
- **Controle de Acesso (RBAC)**: Implementação de rotas exclusivas para administradores (Painel de Monitoramento) e rotas padrão para analistas (Predição em Lote).
- **Banco de Dados de Alta Performance**: Utilização de SQLAlchemy com `bulk_insert_mappings` para registrar milhares de predições simultâneas em milissegundos, viabilizando o histórico completo.
- **Monitoramento de Data Drift**: Rota de auditoria que utiliza o Teste de Kolmogorov-Smirnov (KS) para comparar a distribuição estatística dos dados em produção com a base de treino original.
- **Frontend Interativo (Streamlit)**: Interface web amigável para upload de planilhas (Excel), download de resultados (CSV) e renderização de gráficos complexos (Plotly) para análise de Drift.
- **MLflow Integrado**: Versionamento do XGBoost, rastreio de hiperparâmetros (via `RandomizedSearchCV`) e salvamento de artefatos visuais (Matriz de Confusão e Feature Importance) diretamente no servidor.
- **Segurança de Configuração**: Uso de variáveis de ambiente (`.env`) para injeção de chaves JWT, parâmetros do banco de dados e URLs da API.
---

## 👥 Contribuição

Pull requests são bem-vindos! Abra uma issue ou contribua diretamente via fork + PR.

---

## 📃 Licença

Este projeto está licenciado sob os termos da licença [MIT](LICENSE).
