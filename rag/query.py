import json, sqlite3, requests, math
import os, argparse, glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORES_DIR = os.path.join(BASE_DIR, "stores")
OLLAMA_URL = os.getenv("VETO_OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("VETO_EMBED_MODEL", "nomic-embed-text")
MODEL_CORE = os.getenv("VETO_CHAT_MODEL", "veto_core")

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)

def embed(text):
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "input": text},
            timeout=120
        )
    except requests.exceptions.RequestException as ex:
        raise RuntimeError(f"Impossible de joindre Ollama sur {OLLAMA_URL}. Demarre Ollama puis reessaie.") from ex
    r.raise_for_status()
    js = r.json()
    if "embedding" in js:
        return js["embedding"]
    if "data" in js and js["data"]:
        return js["data"][0]["embedding"]
    if "error" in js:
        raise RuntimeError(f"Ollama embeddings error: {js['error']}")
    raise RuntimeError(f"Unexpected embeddings schema: {js}")

def list_rag_stores():
    stores = {}
    if os.path.isdir(STORES_DIR):
        for path in glob.glob(os.path.join(STORES_DIR, "*.sqlite")):
            rag = os.path.splitext(os.path.basename(path))[0]
            stores[rag] = path

    return stores

def search(query, k=6, rag=None):
    stores = list_rag_stores()
    if rag:
        if rag not in stores:
            raise RuntimeError(f"RAG '{rag}' introuvable. Disponibles: {', '.join(sorted(stores.keys()))}")
        stores = {rag: stores[rag]}

    if not stores:
        raise RuntimeError("Aucun index trouve. Lance d'abord index.py")

    qv = embed(query)
    scored = []

    for rag_name, db_path in stores.items():
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT path, text, embedding FROM chunks")
        rows = cur.fetchall()
        con.close()

        for path, text, emb_json in rows:
            v = json.loads(emb_json)
            scored.append({
                "rag": rag_name,
                "path": path,
                "text": text,
                "score": cosine(qv, v)
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]

def ask_vetia(question, rag=None, k=6):
    hits = search(question, k=k, rag=rag)
    context_lines = []
    for i, hit in enumerate(hits, start=1):
        snippet = hit["text"].strip().replace("\n", " ")[:500]
        context_lines.append(f"[{i}] ({hit['rag']}) {os.path.basename(hit['path'])} :: {snippet}")

    context = "\n".join(context_lines)

    prompt = f"""
[CONTEXTE - Extraits numerotes]
{context}

[QUESTION]
{question}

[INSTRUCTION]
Tu es VetIA. Reponds en t'appuyant strictement sur le CONTEXTE ci-dessus.
- Cite entre crochets les numeros des extraits (ex: [1][3]) quand tu utilises une info.
- Si une information n'est pas dans le contexte, dis-le explicitement.
- Ne pose jamais de diagnostic definitif.
- Si un signe de gravite est present, recommande une urgence veterinaire immediate.
- Reponse en 5 sections maximum:
    1) Niveau estime (faible/moderee/elevee)
    2) Pourquoi
    3) Action immediate
    4) Questions prioritaires
    5) Avertissement
"""

    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL_CORE, "prompt": prompt},
        timeout=180, stream=True
    )
    r.raise_for_status()
    full = ""
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line.decode())
        if "response" in data:
            full += data["response"]
    return full.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Question-reponse sur un ou plusieurs RAG")
    parser.add_argument("--rag", type=str, default=None, help="Nom du RAG cible (sinon recherche globale)")
    parser.add_argument("--k", type=int, default=6, help="Nombre d'extraits recuperes")
    parser.add_argument("--list-rags", action="store_true", help="Affiche les RAG disponibles")
    parser.add_argument("question", nargs="*", help="Question (optionnel: mode interactif si absent)")
    args = parser.parse_args()

    if args.list_rags:
        stores = list_rag_stores()
        if not stores:
            print("Aucun RAG indexe pour le moment.")
        else:
            print("RAG disponibles:")
            for name in sorted(stores.keys()):
                print(f"- {name}")
        raise SystemExit(0)

    question = " ".join(args.question).strip() if args.question else ""
    if not question:
        question = input("Question : ")

    print(ask_vetia(question, rag=args.rag, k=args.k))