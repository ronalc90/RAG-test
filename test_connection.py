# test_connection.py  — MODO DSN (recomendado)
import os
from dotenv import load_dotenv
import pyodbc

load_dotenv()

DSN  = os.getenv("IRIS_DSN", "IRIS")        # p.ej. IRIS
USER = os.getenv("IRIS_USER", "_SYSTEM")
PWD  = os.getenv("IRIS_PASSWORD", "")

print("Drivers ODBC instalados:", pyodbc.drivers())

conn_str = f"DSN={DSN};UID={USER};PWD={PWD}"
print("Probando conexión con:", conn_str)

try:
    with pyodbc.connect(conn_str, timeout=5, autocommit=True) as con:
        cur = con.cursor()
        cur.execute("SELECT 1")
        val = cur.fetchone()[0]
        print("✅ Conectado. SELECT 1 ->", val)
    print("✅ Cierre correcto.")
except Exception as e:
    print("❌ Error de conexión por DSN:", e)
