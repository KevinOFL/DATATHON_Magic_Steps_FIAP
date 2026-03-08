from datetime import datetime, timedelta
from jose import jwt
import os
from dotenv import load_dotenv

load_dotenv()


def create_access_token(data: dict):
    """
    Gera um token JWT de acesso criptografado com tempo de expiração definido.

    Recebe um payload de dados (geralmente contendo a identificação do usuário), 
    adiciona uma data de expiração calculada a partir do tempo atual em UTC 
    mais os minutos definidos nas variáveis de ambiente, e codifica essas 
    informações em uma string JWT utilizando a chave secreta e o algoritmo 
    configurados no sistema.

    parametros:
    - data (dict): Dicionário contendo os dados a serem embutidos no payload do token (ex: {"sub": "email@dominio.com"}).

    retorno:
    - str: O token JWT codificado em formato de string, pronto para ser enviado ao cliente.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, os.getenv("SECRET_KEY"), algorithm=os.getenv("ALGORITHM"))
    return encoded_jwt