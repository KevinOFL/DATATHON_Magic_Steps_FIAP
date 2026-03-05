from sqlalchemy import Column, String, Date, Integer
from werkzeug.security import generate_password_hash, check_password_hash
from app.config.database_config import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(64), nullable=False)
    last_name = Column(String(128), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    position = Column(String(64), nullable=False)

    @property
    def password(self):
        """
        Impede que a senha seja lida diretamente no código.
        """
        raise AttributeError('A senha não pode ser lida diretamente.')

    @password.setter
    def password(self, plain_text_password):
        """
        Intercepta a atribuição (ex: user.password = "123")
        e salva o hash no banco de dados.
        """
        self.password_hash = generate_password_hash(plain_text_password)

    def verify_password(self, plaintext_password):
        """
        Método para verificar se uma senha digitada no login
        corresponde ao hash salvo no banco de dados.
        """
        return check_password_hash(self.password_hash, plaintext_password)
