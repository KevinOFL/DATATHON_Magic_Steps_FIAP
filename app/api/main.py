import io
import mlflow
import pandas as pd
import numpy as np
import os
from scipy.stats import ks_2samp
from datetime import datetime, timedelta
from dotenv import load_dotenv
from jose import JWTError, jwt
from fastapi import FastAPI, File, HTTPException, UploadFile, status, Depends
from fastapi.responses import Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from app.config.log_config import logger_api, logger_db
from app.src.data_loader import loading_data_prediction, loading_data
from sqlalchemy.orm import Session
from app.config.database_config import SessionLocal, engine
from app.models.user_model import User
from app.models.previsions_model import PredicitonRequest
from app.models.data_previsions_model import DataPrevisions
from app.models import user_model
from app.models import previsions_model
from app.models import data_previsions_model
from app.schemas.user_schema import UserCreate
from app.api.security import create_access_token

load_dotenv()

user_model.Base.metadata.create_all(bind=engine)
previsions_model.Base.metadata.create_all(bind=engine)
data_previsions_model.Base.metadata.create_all(bind=engine)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Inicializa a API
app = FastAPI(
    title="API de Predição de Evasão",
    description="API para classificar o risco de evasão de alunos (Datathon)",
    version="1.0",
)


def get_db():
    """
    Gerencia o ciclo de vida de uma sessão do banco de dados para cada requisição.

    Cria uma nova instância de sessão (SessionLocal), fornece essa sessão para
    uso nas operações de banco de dados (via yield) e garante, através de um
    bloco try/finally, que a conexão seja sempre fechada ao final da requisição,
    registrando todas as etapas (início, sucesso e encerramento) nos logs.

    parametros:
    - Nenhum.

    retorno:
    - Session: Uma instância ativa da sessão do banco de dados (gerada via yield).
    """
    db = SessionLocal()
    logger_db.info("Sessão do banco de dados iniciada")
    try:
        yield db
        logger_db.info("Sessão do banco de dados finalizada com sucesso")
    finally:
        logger_db.info("Fechando sessão do banco de dados")
        db.close()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Valida o token JWT da requisição e recupera o usuário autenticado atual.

    Decodifica o token fornecido utilizando as credenciais do ambiente (SECRET_KEY e ALGORITHM),
    extrai o e-mail (armazenado na chave 'sub') e consulta o banco de dados. Caso o token
    seja inválido, não possua e-mail ou o usuário não seja encontrado, levanta uma 
    exceção HTTP 401 (Não Autorizado).

    parametros:
    - token (str): O token JWT extraído do cabeçalho de autorização da requisição.
    - db (Session): A sessão ativa do banco de dados (injetada como dependência).

    retorno:
    - User: A instância do modelo de usuário correspondente ao e-mail autenticado.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Não foi possível validar as credenciais.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            os.getenv("SECRET_KEY"),
            algorithms=[os.getenv("ALGORITHM")]
        )
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


async def get_current_admin_user(current_user: User = Depends(get_current_user)):
    """
    Valida se o usuário autenticado possui privilégios de administrador.

    Atua como uma dependência de autorização de rotas no FastAPI, estendendo 
    a validação base do token JWT (get_current_user). Verifica a propriedade 
    'position' do perfil do usuário. Caso o cargo não seja "admin" (ignorando 
    maiúsculas/minúsculas) ou seja nulo, barra a requisição levantando uma 
    exceção HTTP 403 (Proibido). Se autorizado, repassa a instância do usuário 
    para o endpoint.

    parametros:
    - current_user (User): A instância do usuário logado, previamente validada e injetada pela dependência get_current_user.

    retorno:
    - User: A instância do modelo de usuário logado e confirmado com cargo de administrador.
    """
    if not current_user.position or current_user.position.lower() != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acesso negado. Esta operação exige privilégios de administrador."
        )
    return current_user


@app.get("/")
async def read_root():
    """
    Endpoint raiz de verificação de disponibilidade (Health Check) da API.

    Retorna uma mensagem simples em formato JSON para confirmar que o servidor
    FastAPI está online, rodando e pronto para receber requisições.

    parametros:
    - Nenhum.

    retorno:
    - dict: Um dicionário contendo a chave "status" e a mensagem de confirmação.
    """
    return {"status": "A API está viva e rodando!"}


@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    """
    Registra um novo usuário no sistema.

    Verifica se o e-mail fornecido já existe no banco de dados. Caso exista, 
    levanta uma exceção HTTP 400. Caso contrário, cria uma nova instância do 
    modelo User, aplica as informações recebidas, salva no banco de dados 
    e retorna uma mensagem de sucesso.

    parametros:
    - user_in (UserCreate): Objeto Pydantic contendo os dados de criação do usuário (nome, e-mail, senha, etc.).
    - db (Session): A sessão ativa do banco de dados (injetada como dependência).

    retorno:
    - dict: Um dicionário contendo a mensagem de confirmação de criação do usuário.
    """
    user_exist = db.query(User).filter(User.email == user_in.email).first()
    if user_exist:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Este e-mail já está cadastrado."
        )
    new_user = User(
        name=user_in.name,
        last_name=user_in.last_name,
        email=user_in.email,
        password=user_in.password,
        date_of_birth=user_in.date_of_birth,
        position=user_in.position
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": f"Usuário {new_user.name} criado com sucesso!"}


@app.post("/login")
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Autentica um usuário e gera um token JWT de acesso.

    Recebe as credenciais em formato de formulário (Form Data), busca o usuário 
    pelo e-mail e verifica se a senha criptografada corresponde. Se as credenciais 
    forem inválidas, levanta uma exceção HTTP 401. Se forem válidas, gera e 
    retorna um token JWT assinado para acesso às rotas protegidas.

    parametros:
    - form_data (OAuth2PasswordRequestForm): Formulário contendo as credenciais 'username' (e-mail) e 'password'.
    - db (Session): A sessão ativa do banco de dados (injetada como dependência).

    retorno:
    - dict: Um dicionário contendo o 'access_token' gerado e o 'token_type' (bearer).
    """
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not user.verify_password(form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="E-mail ou palavra-passe incorretos.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/me")
async def read_user_me(current_user: User = Depends(get_current_user)):
    """
    Retorna os dados de perfil do usuário atualmente autenticado.

    Utiliza a dependência get_current_user para validar o token JWT enviado no 
    cabeçalho da requisição. Se o token for válido, extrai e retorna as informações 
    públicas do perfil do usuário logado.

    parametros:
    - current_user (User): A instância do usuário logado validado (injetada como dependência).

    retorno:
    - dict: Um dicionário contendo uma mensagem de sucesso, nome, e-mail e cargo do usuário.
    """
    return {
        "message": "Acesso autorizado!",
        "name": current_user.name,
        "email": current_user.email,
        "position": current_user.position
    }


@app.post("/predict/batch")
async def predict_droput(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Processa um arquivo Excel para predição de risco de evasão escolar.

    Valida o formato do arquivo, registra a requisição no banco de dados e carrega 
    a versão mais recente do modelo XGBoost diretamente do MLflow. Lê os dados 
    utilizando Pandas, aplica a régua de corte configurada (0.35) e adiciona as 
    colunas de probabilidade e alerta no DataFrame original. Por fim, formata 
    os dados e devolve um arquivo CSV pronto para download.

    parametros:
    - file (UploadFile): O arquivo de planilha carregado pelo usuário contendo os dados dos alunos.
    - current_user (User): A instância do usuário logado que solicitou a predição (injetada como dependência).
    - db (Session): A sessão ativa do banco de dados (injetada como dependência).

    retorno:
    - Response: Uma resposta HTTP configurada para download de arquivo (attachment), contendo os dados processados em formato CSV (text/csv).
    """
    if not file.filename.endswith((".xls", ".xlsx")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Envie apenas planilhas excel. (.xlsx ou .xls)"
        )
    try:
        new_registry = PredicitonRequest(
            file_name=file.filename,
            file_size_bytes=file.size,
            user_id=current_user.id
        )
        db.add(new_registry)
        db.commit()
        db.refresh(new_registry)
        logger_api.info(f"Registro de predição {new_registry.id} salvo para o usuário {current_user.email}.")
        # Aqui podemos atualizar essa variavel baseado na escolha
        # de modelo de corte do cliente
        model_name = "Best_Model_XGBoost_regua_de_corte_0_35"
        logger_api.info(f"Usuário {current_user.email} iniciou predição. Buscando modelo: {model_name}")
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                details="Modelo de predição não encontrado no registro para efetuar a requisição.\nTente novamente mais tarde.",
            )
        # Efetuo um sort para pegar o modelo da versão atual
        versions.sort(key=lambda x: int(x.version))
        latest_version_obj = versions[-1]
        version_number = latest_version_obj.version
        current_stage = latest_version_obj.current_stage
        model_uri = f"models:/{model_name}/{version_number}"
        logger_api.info(
            f"Carregando versão {version_number} (Stage: {current_stage}) do URI: {model_uri}"
        )
        model = mlflow.sklearn.load_model(model_uri=model_uri)
        logger_api.info("Modelo carregado com sucesso...")

        contents = await file.read()
        df_predict = pd.read_excel(io.BytesIO(contents))
        # Faço uma cópia do arquivo original pois como ele e processado
        # para uma visualização do modelo o cliente pode perde
        # o rastreamento de X aluno
        df_original = pd.read_excel(io.BytesIO(contents))

        if df_original.empty:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Arquivo enviado está inválido ou vazio.\nPor favor re-envie um arquivo válido.",
            )
        # Efetuo a predição com o modelo selecionado
        df_predict = loading_data_prediction(df_predict.copy())
        # Ajusto a ordem das colunas
        df_predict = df_predict[
            [
                "idade",
                "genero",
                "ano_ingresso",
                "mat",
                "diferenca_mat",
                "por",
                "diferenca_por",
                "ing",
                "diferenca_ing",
                "estudou_ingles",
                "iaa",
                "ieg",
                "diferenca_ieg",
                "ips",
                "ida",
                "ipv",
                "ian",
                "inde",
                "diferenca_inde",
            ]
        ]

        logger_api.info("Iniciando o cálculo das predições...")
        # Pego a probabilidade de evasão que o modelo entregou
        probabilidades = model.predict_proba(df_predict)[:, 1]
        probabilidades_percentuais = (probabilidades * 100).round(0).astype(int)
        # Aplico a regra de corte e efetuo o calculo de quem pode evadir ou não
        regua_corte = 0.35
        previsoes = (probabilidades >= regua_corte).astype(int)
        # Entrego esses dados a uma coluna no dataframe do arquivo original
        df_original["probabilidade_risco_%"] = probabilidades_percentuais
        df_original["alerta_evasao"] = previsoes
        logger_api.info("Predições finalizadas com sucesso!")
        logger_api.info("Salvando histórico de predições no banco de dados...")
        df_for_db = df_predict.copy()
        df_for_db["request_id"] = new_registry.id
        df_for_db["probabilidade_risco_percentual"] = probabilidades_percentuais
        df_for_db["alerta_evasao"] = previsoes
        registros_dict = df_for_db.replace({float('nan'): None}).to_dict(orient="records")
        db.bulk_insert_mappings(DataPrevisions, registros_dict)
        db.commit()
        logger_api.info(f"{len(registros_dict)} alunos salvos no histórico com sucesso!")
        csv_data = df_original.to_csv(
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8"
        )
        content_with_bom = "\ufeff" + csv_data
        # Retorno o arquivo original com as novas colunas no tipo
        # .csv por ser mais leve
        return Response(
            content=content_with_bom,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename=previsoes_evasao_risco_{regua_corte}.csv"
            },
        )
    except Exception as e:
        logger_api.error(f"Erro interno: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/monitoring/drift", tags={"Administração"})
async def calculate_data_drift(
  feature: str,
  history_day: int = 30,
  admin_user: User = Depends(get_current_admin_user),
  db: Session = Depends(get_db)
):
    """
    Calcula o Data Drift comparando dados de produção recentes com a base de treinamento.

    Esta rota é restrita a administradores. Ela recebe o nome de uma variável (feature), 
    consulta no banco de dados as predições reais feitas nos últimos 'X' dias e compara 
    essa distribuição com a base original de treinamento utilizando o Teste Estatístico 
    Kolmogorov-Smirnov (KS). Retorna se há drift, o P-Value e os dados para plotagem 
    do histograma comparativo no front-end.

    parametros:
    - feature (str): O nome exato da coluna a ser analisada (ex: 'inde', 'idade').
    - dias_historico (int): A janela de tempo em dias para buscar dados de produção (padrão: 30).
    - current_user (User): O usuário autenticado (injetado via dependência).
    - db (Session): A sessão ativa do banco de dados (injetada via dependência).

    retorno:
    - dict: Um JSON contendo o status do drift, score estatístico e os arrays para o gráfico.
    """
    exclude_collumns = [
            "ra",
            "fase",
            "turma",
            "nome",
            "ano_nasc",
            "instituicao_de_ensino",
            "pedra_20",
            "pedra_21",
            "pedra_22",
            "cg",
            "cf",
            "ct",
            "no_av",
            "avaliador1",
            "rec_av1",
            "avaliador2",
            "rec_av2",
            "avaliador3",
            "rec_av3",
            "avaliador4",
            "rec_av4",
            "rec_psicologia",
            "indicado",
            "atingiu_pv",
            "fase_ideal",
            "destaque_ieg",
            "destaque_ida",
            "destaque_ipv",
            "defas",
        ]
    logger_api.info(f"O administrador {admin_user.email} acessou o painel de monitoramento.")
    try:
        # Pego a mesma base de dados que foi usada nos treinos
        df = pd.read_excel("app/data/base_de_dados_pede_2024.xlsx", sheet_name="PEDE2022")
        df_referencia = loading_data(df, exclude_collumns)
    except Exception as e:
        logger_api.warning(f"Aviso: Base de referência não encontrada. O Drift não funcionará. Erro: {e}")
        df_referencia = pd.DataFrame()

    if df_referencia.empty:
        raise HTTPException(status_code=500, detail="Base de referência não carregada no servidor.")

    if feature not in df_referencia.columns:
        raise HTTPException(status_code=400, detail=f"A variável '{feature}' não existe na base de treinamento.")
    try:
        limit_date = datetime.utcnow() - timedelta(days=history_day)
        db_collum = getattr(DataPrevisions, feature)
        db_result = db.query(db_collum).filter(
            DataPrevisions.created_at >= limit_date
        ).all()
        if not db_result:
            raise HTTPException(
                status_code=404,
                detail=f"Não há predições nos últimos {history_day} dias para calcular o drift."
            )
        curr_data = np.array([res[0] for res in db_result if res[0] is not None])
        ref_data = df_referencia[feature].dropna().values
        if len(curr_data) < 10:
            raise HTTPException(
                 status_code=400,
                 detail="Dados insuficientes na produção para um cálculo estatístico confiável."
                )
        statistic, p_value = ks_2samp(ref_data, curr_data)
        drift_detected = bool(p_value < 0.05)
        bins = np.histogram_bin_edges(np.concatenate([ref_data, curr_data]), bins=10)
        hist_ref, _ = np.histogram(ref_data, bins=bins)
        hist_curr, _ = np.histogram(curr_data, bins=bins)

        categorias_x = [f"{bins[i]:.1f} a {bins[i+1]:.1f}" for i in range(len(bins)-1)]
        return {
            "feature": feature,
            "dias_analisados": history_day,
            "amostras_producao": len(curr_data),
            "drift_detected": drift_detected,
            "drift_score_ks": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "grafico_dados": {
                "eixo_x_categorias": categorias_x,
                "distribuicao_referencia": [int(x) for x in hist_ref],
                "distribuicao_atual": [int(x) for x in hist_curr]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger_api.error(f"Erro ao calcular Drift: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao calcular drift: {str(e)}")