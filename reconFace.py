# =====================================
# Importação de bibliotecas
# =====================================

from flask import Flask, request, jsonify  
import psycopg2  
import cv2  
import face_recognition  
import numpy as np 
import redis  
import requests  
from dotenv import load_dotenv 
import os 

load_dotenv()  
# =============================================
# Conexão com os bancos de dados (SQL e Redis)
# =============================================

conn_sql = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    database=os.getenv("PG_DATABASE"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    port=int(os.getenv("PG_PORT"))
)
cursor = conn_sql.cursor()  # cursor para executar comandos SQL

r = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    db=int(os.getenv("REDIS_DB"))
)

app = Flask(__name__)  # cria o servidor Flask


# ===========================
# Funções
# ===========================

# função para pegar imagem do Redis e converter para formato OpenCV
def carregar_imagem_redis(chave: str):
    img_bytes = r.get(chave)  # pega os bytes da imagem
    if not img_bytes:
        return None
    np_arr = np.frombuffer(img_bytes, np.uint8)  # converte bytes para array
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # transforma array em imagem OpenCV

# função para baixar imagem de uma URL e salvar no Redis
def baixar_e_salvar_imagem(url: str, chave: str):
    try:
        resp = requests.get(url, timeout=5)  # baixa a imagem
        if resp.status_code == 200:
            r.set(chave, resp.content)  # salva no Redis
            return True
        return False
    except Exception:
        return False
    


# ======================================
# Rota de verificação facial
# ======================================

# rota para verificar se a foto enviada bate com a foto de referência
@app.route("/verificar_face", methods=["POST"])
def verificar_face():
    try:
        # pega os dados enviados pelo app
        user_id = request.form.get("user_id")
        foto_teste = request.files.get("foto_teste")

        # verifica se os dados existem
        if not user_id or not foto_teste:
            return jsonify({"resultado": False, "erro": "Parâmetros inválidos"}), 400

        user_id = int(user_id)  # transforma o ID em inteiro

        # salva a foto de teste no Redis temporariamente
        foto_bytes = foto_teste.read()
        chave_teste = f"foto_teste_{user_id}"
        r.set(chave_teste, foto_bytes, ex=30)  # expira em 30 segundos

        # tenta pegar a imagem de referência do Redis
        chave_ref = f"foto_referencia_{user_id}"
        img_ref = carregar_imagem_redis(chave_ref)

        if img_ref is None:
            # se não estiver no Redis, pega do banco
            cursor.execute("SELECT imagem_url FROM operario WHERE id = %s", (user_id,))
            resultado = cursor.fetchone()
            if not resultado:
                return jsonify({"resultado": False, "erro": "Usuário sem imagem de referência."})

            url_ref = resultado[0]
            # baixa a imagem e salva no Redis
            if not baixar_e_salvar_imagem(url_ref, chave_ref):
                return jsonify({"resultado": False, "erro": "Erro ao baixar imagem de referência."})

            img_ref = carregar_imagem_redis(chave_ref)  # carrega do Redis novamente

        # carrega a imagem de teste do Redis
        img_teste = carregar_imagem_redis(chave_teste)
        if img_teste is None:
            return jsonify({"resultado": False, "erro": "Erro ao processar imagem enviada."})

        # converte imagens para RGB (necessário para face_recognition)
        img_ref_rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
        img_teste_rgb = cv2.cvtColor(img_teste, cv2.COLOR_BGR2RGB)

        # gera os encodings (vetores) dos rostos
        ref_enc = face_recognition.face_encodings(img_ref_rgb)
        test_enc = face_recognition.face_encodings(img_teste_rgb)

        # se não detectou rosto em alguma imagem
        if not ref_enc or not test_enc:
            return jsonify({"resultado": False, "erro": "Rosto não detectado em uma das imagens."})

        # compara os rostos
        match = face_recognition.compare_faces([ref_enc[0]], test_enc[0])[0]

        return jsonify({"resultado": bool(match)})  # retorna True ou False

    except Exception as e:
        return jsonify({"resultado": False, "erro": str(e)}), 500  # retorna erro se algo falhar


