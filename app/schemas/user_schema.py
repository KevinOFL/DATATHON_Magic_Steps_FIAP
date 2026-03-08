from pydantic import BaseModel, EmailStr
from datetime import date


# Esquema para o Registro (exige todos os campos da sua model)
class UserCreate(BaseModel):
    name: str
    last_name: str
    email: EmailStr
    password: str
    date_of_birth: date
    position: str
