import os, sqlite3, json, hashlib, time, argparse
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", "kb"))
STORES_DIR = os.path.join(BASE_DIR, "stores")
EMBED_MODEL = os.getenv("VETO_EMBED_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("VETO_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("VETO_CHUNK_OVERLAP", "200"))
BATCH_SIZE = int(os.getenv("VETO_BATCH_SIZE", "8"))
OLLAMA_URL = os.getenv("VETO_OLLAMA_URL", "http://localhost:11434")

def db_path_for_rag(rag_name):
    safe = rag_name.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    if not safe:
        safe = "default"
    os.makedirs(STORES_DIR, exist_ok=True)
    return os.path.join(STORES_DIR, f"{safe}.sqlite")

def ensure_db(db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        path TEXT,
        idx INTEGER,
        text TEXT,
        embedding BLOB
    )
    """)
    con.commit()
    con.close()

def read_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(t):
    chunks = []
    i = 0
    while i < len(t):
        chunks.append(t[i:i+CHUNK_SIZE])
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def embed_batch(texts):
    """Tente un batch embeddings. Supporte plusieurs schémas de réponse.
       Fallback en unitaire si nécessaire."""
    url = f"{OLLAMA_URL}/api/embeddings"
    payload = {"model": EMBED_MODEL, "input": texts}
    r = requests.post(url, json=payload, timeout=180)
    # Si l’API renvoie une erreur HTTP, lève l’exception explicite
    try:
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"HTTP error from embeddings endpoint: {r.status_code} {r.text}") from e

    try:
        js = r.json()
    except Exception as e:
        raise RuntimeError(f"Invalid JSON from embeddings endpoint: {r.text}") from e

    # Cas 1: format batch: {"data":[{"embedding":[...]}, ...]}
    if isinstance(js, dict) and "data" in js and isinstance(js["data"], list):
        return [item["embedding"] for item in js["data"]]

    # Cas 2: certains serveurs ne gèrent pas la liste et renvoient un seul "embedding"
    # -> on retente en unitaire
    if isinstance(texts, list) and len(texts) > 1:
        vectors = []
        for t in texts:
            vectors.append(embed_single(t))
        return vectors

    # Cas 3: format single: {"embedding":[...]}
    if "embedding" in js:
        return [js["embedding"]]

    # Cas 4: message d’erreur applicatif
    if "error" in js:
        raise RuntimeError(f"Ollama embeddings error: {js['error']}")

    raise RuntimeError(f"Unexpected embeddings response schema: {js}")

def embed_single(text):
    url = f"{OLLAMA_URL}/api/embeddings"
    payload = {"model": EMBED_MODEL, "input": text}
    r = requests.post(url, json=payload, timeout=60)
    try:
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"HTTP error (single) {r.status_code} {r.text}") from e

    js = r.json()
    if "embedding" in js:
        return js["embedding"]
    if "data" in js and isinstance(js["data"], list) and js["data"]:
        return js["data"][0]["embedding"]
    if "error" in js:
        raise RuntimeError(f"Ollama embeddings error (single): {js['error']}")
    raise RuntimeError(f"Unexpected single embeddings schema: {js}")

def list_files_grouped_by_rag():
    grouped = {}
    if not os.path.isdir(KB_ROOT):
        return grouped

    for root, _, files in os.walk(KB_ROOT):
        for name in files:
            if not name.lower().endswith((".txt", ".md")):
                continue

            path = os.path.join(root, name)
            rel = os.path.relpath(path, KB_ROOT)
            parts = rel.split(os.sep)
            rag_name = parts[0] if len(parts) > 1 else "default"
            grouped.setdefault(rag_name, []).append(path)

    return grouped

def index_rag(rag_name, file_paths, rebuild=False):
    db_path = db_path_for_rag(rag_name)
    ensure_db(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    if rebuild:
        cur.execute("DELETE FROM chunks")
        con.commit()

    indexed = 0
    for path in file_paths:
        text = read_file(path)
        parts = chunk_text(text)

        vectors = []
        for i in range(0, len(parts), BATCH_SIZE):
            batch = parts[i:i+BATCH_SIZE]
            try:
                vectors.extend(embed_batch(batch))
            except Exception as e:
                print(f"[WARN] batch embeddings failed ({e}), falling back to single...")
                for ch in batch:
                    try:
                        vectors.append(embed_single(ch))
                        time.sleep(0.05)
                    except Exception as e2:
                        print(f"[ERROR] embedding failed on chunk: {e2}")
                        vectors.append([0.0]*768)

        for i, (chunk, vec) in enumerate(zip(parts, vectors)):
            _id = hashlib.sha256(f"{path}:{i}".encode()).hexdigest()
            cur.execute("""
                INSERT OR REPLACE INTO chunks (id,path,idx,text,embedding)
                VALUES (?,?,?,?,?)
            """, (_id, path, i, chunk, json.dumps(vec)))
            indexed += 1

    con.commit()
    con.close()
    print(f"[{rag_name}] Indexation terminée. {indexed} chunks enregistrés dans {db_path}.")

def index_kb(target_rag=None, rebuild=False):
    grouped = list_files_grouped_by_rag()
    if not grouped:
        print(f"Aucun fichier .md/.txt trouvé dans {KB_ROOT}")
        return

    if target_rag:
        if target_rag not in grouped:
            print(f"RAG '{target_rag}' introuvable. RAG disponibles: {', '.join(sorted(grouped.keys()))}")
            return
        index_rag(target_rag, grouped[target_rag], rebuild=rebuild)
        return

    for rag_name in sorted(grouped.keys()):
        index_rag(rag_name, grouped[rag_name], rebuild=rebuild)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexe un ou plusieurs RAG depuis le dossier kb/")
    parser.add_argument("--rag", type=str, default=None, help="Nom du RAG (nom du sous-dossier dans kb/)")
    parser.add_argument("--rebuild", action="store_true", help="Reconstruit l'index du/des RAG ciblé(s)")
    args = parser.parse_args()
    index_kb(target_rag=args.rag, rebuild=args.rebuild)