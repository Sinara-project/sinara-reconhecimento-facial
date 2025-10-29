# ======================================
# reconFace.py — Módulo de Reconhecimento Facial
# ======================================

from flask import Blueprint, request, jsonify
import psycopg2
import cv2
import face_recognition
import numpy as np
import redis
import requests
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente
load_dotenv()

# ======================================
# Conexões com bancos de dados
# ======================================
conn_sql = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    database=os.getenv("PG_DATABASE"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    port=int(os.getenv("PG_PORT"))
)
cursor = conn_sql.cursor()

r = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    db=int(os.getenv("REDIS_DB"))
)

# ======================================
# Criação da Blueprint
# ======================================
reconface_bp = Blueprint("reconface_bp", __name__)

# ======================================
# Funções auxiliares
# ======================================

def carregar_imagem_redis(chave: str):
    """Pega imagem armazenada no Redis e converte em formato OpenCV."""
    img_bytes = r.get(chave)
    if not img_bytes:
        return None
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def baixar_e_salvar_imagem(url: str, chave: str):
    """Baixa imagem de uma URL e salva no Redis."""
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            r.set(chave, resp.content)
            return True
        return False
    except Exception:
        return False

# ======================================
# Rota de verificação facial
# ======================================
@reconface_bp.route("/verificar_face", methods=["POST"])
def verificar_face():
    """Verifica se a foto enviada bate com a imagem de referência."""
    try:
        user_id = request.form.get("user_id")
        foto_teste = request.files.get("foto_teste")

        if not user_id or not foto_teste:
            return jsonify({"resultado": False, "erro": "Parâmetros inválidos"}), 400

        user_id = int(user_id)
        foto_bytes = foto_teste.read()
        chave_teste = f"foto_teste_{user_id}"
        r.set(chave_teste, foto_bytes, ex=30)

        chave_ref = f"foto_referencia_{user_id}"
        img_ref = carregar_imagem_redis(chave_ref)

        if img_ref is None:
            cursor.execute("SELECT imagem_url FROM operario WHERE id = %s", (user_id,))
            resultado = cursor.fetchone()
            if not resultado:
                return jsonify({"resultado": False, "erro": "Usuário sem imagem de referência."})

            url_ref = resultado[0]
            if not baixar_e_salvar_imagem(url_ref, chave_ref):
                return jsonify({"resultado": False, "erro": "Erro ao baixar imagem de referência."})

            img_ref = carregar_imagem_redis(chave_ref)

        img_teste = carregar_imagem_redis(chave_teste)
        if img_teste is None:
            return jsonify({"resultado": False, "erro": "Erro ao processar imagem enviada."})

        img_ref_rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
        img_teste_rgb = cv2.cvtColor(img_teste, cv2.COLOR_BGR2RGB)

        ref_enc = face_recognition.face_encodings(img_ref_rgb)
        test_enc = face_recognition.face_encodings(img_teste_rgb)

        if not ref_enc or not test_enc:
            return jsonify({"resultado": False, "erro": "Rosto não detectado em uma das imagens."})

        match = face_recognition.compare_faces([ref_enc[0]], test_enc[0])[0]
        return jsonify({"resultado": bool(match)})

    except Exception as e:
        return jsonify({"resultado": False, "erro": str(e)}), 500
