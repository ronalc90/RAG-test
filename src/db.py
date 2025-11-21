import pyodbc
from .config import IRIS_DSN, IRIS_USER, IRIS_PASSWORD

def get_conn():
    # Requiere que tengas configurado el DSN de InterSystems IRIS en Windows
    return pyodbc.connect(f"DSN={IRIS_DSN};UID={IRIS_USER};PWD={IRIS_PASSWORD}")
