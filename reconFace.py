# ======================================
# reconFace.py — Módulo de Reconhecimento Facial (otimizado)
# ======================================

from flask import Blueprint, request, jsonify
import psycopg2
import redis
import requests
from dotenv import load_dotenv
import os
import io

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

r = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    db=int(os.getenv("REDIS_DB"))
)

reconface_bp = Blueprint("reconface_bp", __name__)

# ======================================
# Imports pesados — lazy load cache
# ======================================
_cv2 = None
_fr = None
_np = None

def get_libs():
    """Importa bibliotecas pesadas apenas uma vez (lazy load)."""
    global _cv2, _fr, _np
    if _cv2 is None or _fr is None or _np is None:
        import cv2, face_recognition, numpy as np
        _cv2, _fr, _np = cv2, face_recognition, np
    return _cv2, _fr, _np

# ======================================
# Funções auxiliares
# ======================================

def carregar_imagem_redis(chave: str):
    """Pega imagem armazenada no Redis e converte em formato OpenCV."""
    cv2, _, np = get_libs()
    img_bytes = r.get(chave)
    if not img_bytes:
        return None
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def baixar_e_salvar_imagem(url: str, chave: str):
    """Baixa imagem e salva no Redis em formato JPEG comprimido."""
    try:
        cv2, _, np = get_libs()
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            ok, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                r.set(chave, buffer.tobytes())
                return True
        return False
    except Exception:
        return False


def get_face_encoding(img, cache_key=None):
    """Gera encoding facial e salva opcionalmente no Redis."""
    _, fr, np = get_libs()
    rgb = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
    encs = fr.face_encodings(rgb)
    if not encs:
        return None
    encoding = encs[0]
    if cache_key:
        r.set(cache_key, np.array(encoding).tobytes())
    return encoding


def get_cached_encoding(chave: str):
    """Recupera encoding armazenado no Redis (caso exista)."""
    _, _, np = get_libs()
    data = r.get(chave)
    if not data:
        return None
    return np.frombuffer(data, dtype=np.float64)

# ======================================
# Rota de verificação facial
# ======================================

@reconface_bp.route("/verificar_face", methods=["POST"])
def verificar_face():
    """Verifica se a foto enviada bate com a imagem de referência."""
    try:
        cv2, fr, np = get_libs()

        user_id = request.form.get("user_id")
        foto_teste = request.files.get("foto_teste")

        if not user_id or not foto_teste:
            return jsonify({"resultado": False, "erro": "Parâmetros inválidos"}), 400

        user_id = int(user_id)
        foto_bytes = foto_teste.read()
        chave_teste = f"foto_teste_{user_id}"
        r.set(chave_teste, foto_bytes, ex=30)

        # --- Encoding de referência ---
        chave_ref = f"foto_referencia_{user_id}"
        chave_ref_enc = f"encoding_ref_{user_id}"

        encoding_ref = get_cached_encoding(chave_ref_enc)
        if encoding_ref is None:
            img_ref = carregar_imagem_redis(chave_ref)
            if img_ref is None:
                with conn_sql.cursor() as cur:
                    cur.execute("SELECT imagem_url FROM operario WHERE id = %s", (user_id,))
                    res = cur.fetchone()
                if not res:
                    return jsonify({"resultado": False, "erro": "Usuário sem imagem de referência."})
                url_ref = res[0]
                if not baixar_e_salvar_imagem(url_ref, chave_ref):
                    return jsonify({"resultado": False, "erro": "Erro ao baixar imagem de referência."})
                img_ref = carregar_imagem_redis(chave_ref)

            encoding_ref = get_face_encoding(img_ref, cache_key=chave_ref_enc)
            if encoding_ref is None:
                return jsonify({"resultado": False, "erro": "Rosto não detectado na imagem de referência."})

        # --- Encoding da imagem teste ---
        img_teste = carregar_imagem_redis(chave_teste)
        if img_teste is None:
            return jsonify({"resultado": False, "erro": "Erro ao processar imagem enviada."})

        encoding_teste = get_face_encoding(img_teste)
        if encoding_teste is None:
            return jsonify({"resultado": False, "erro": "Rosto não detectado na imagem enviada."})

        # --- Comparação ---
        match = fr.compare_faces([encoding_ref], encoding_teste, tolerance=0.5)[0]
        return jsonify({"resultado": bool(match)})

    except Exception as e:
        return jsonify({"resultado": False, "erro": str(e)}), 500
