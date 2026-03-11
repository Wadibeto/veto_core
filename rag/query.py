import json, sqlite3, requests, math, re
import os, argparse, glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORES_DIR = os.path.join(BASE_DIR, "stores")
OLLAMA_URL = os.getenv("VETO_OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("VETO_EMBED_MODEL", "nomic-embed-text")
MODEL_CORE = os.getenv("VETO_CHAT_MODEL", "veto_core")
DEFAULT_TOP_K = int(os.getenv("VETO_TOP_K", "4"))
DEFAULT_NUM_PREDICT = int(os.getenv("VETO_NUM_PREDICT", "220"))

LOW_INFO_HINTS = (
    "bonjour", "bonsoir", "salut", "aide", "besoin d'aide", "besoin d aide",
    "mon chat", "mon chien", "aidez-moi", "aidez moi", "help"
)

CLINICAL_KEYWORDS = (
    "vom", "diarr", "respir", "touss", "eternu", "sang", "douleur", "boit", "mange",
    "appetit", "abattu", "fatigue", "urine", "pipi", "temperature", "fievre",
    "plaie", "convulsion", "demange", "gratte", "boite", "bave", "gonf", "constip",
    "tremble", "létharg", "letharg", "agress", "poids", "24h", "48h"
)

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

def search(query, k=DEFAULT_TOP_K, rag=None):
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

def _build_context_lines(hits):
    context_lines = []
    for hit in hits:
        snippet = hit["text"].strip().replace("\n", " ")[:320]
        context_lines.append(f"- {os.path.basename(hit['path'])} ({hit['rag']}): {snippet}")
    return context_lines

def _normalize_history(history):
    normalized = []
    if not history:
        return normalized

    for item in history[-10:]:
        role = (item.get("role") or "").strip().lower()
        content = (item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized

def sanitize_answer_text(text):
    cleaned = text.replace("**", "")
    cleaned = re.sub(r"(?:\[(?:\d+)\])+", "", cleaned)
    cleaned = re.sub(r"\[(?:\d+(?:\s*,\s*\d+)*)\]", "", cleaned)
    cleaned = re.sub(r"^[\-*]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def should_ask_clarifying_questions(question, history=None):
    normalized_question = question.strip().lower()
    if not normalized_question:
        return True

    if any(keyword in normalized_question for keyword in CLINICAL_KEYWORDS):
        return False

    normalized_history = _normalize_history(history)
    if normalized_history:
        history_text = " ".join(item["content"].lower() for item in normalized_history if item["role"] == "user")
        if any(keyword in history_text for keyword in CLINICAL_KEYWORDS):
            return False

    is_short = len(normalized_question) < 90
    looks_generic = any(hint in normalized_question for hint in LOW_INFO_HINTS)
    return is_short and looks_generic

def build_clarification_reply():
    return (
        "Bien sur, je peux vous aider. Pour vous repondre utilement, il me faut d'abord quelques details sur votre chat.\n\n"
        "Pouvez-vous me dire :\n"
        "Quels symptomes vous observez exactement ?\n"
        "Depuis quand cela a commence ?\n"
        "Est-ce qu'il mange, boit et urine normalement ?\n"
        "Est-ce qu'il vomit, a la diarrhee, tousse ou respire mal ?\n"
        "Est-ce qu'il est plutot alerte, fatigue, cache ou douloureux ?\n\n"
        "S'il y a une difficulte a respirer, des vomissements repetes, du sang, une grande faiblesse ou une douleur importante, il faut contacter rapidement un veterinaire."
    )

def build_chat_messages(question, hits, history=None):
    context_lines = _build_context_lines(hits)
    context_block = "\n".join(context_lines) if context_lines else "- Aucun extrait pertinent disponible."

    system_content = f"""
Tu es VetIA, un assistant conversationnel pour une clinique veterinaire.

Tu dois parler comme un assistant de chat moderne, naturel et interactif.
- Commence par repondre de facon humaine et directe.
- Par defaut, reponds comme ChatGPT ou Mistral dans une conversation utile: simple, fluide, naturel.
- Si la demande est vague ou si les symptomes ne sont pas decrits, ne donne pas de niveau de gravite par defaut.
- Dans ce cas, pose d'abord 2 a 5 questions courtes, utiles et tres concretes pour comprendre la situation.
- N'utilise un niveau de gravite que si les informations sont suffisantes.
- Quand tu as assez d'informations, reponds en prose naturelle, concise, sans recracher un cours.
- Tu peux structurer legerement la reponse si c'est utile, mais n'impose pas un gabarit fixe a chaque tour.
- Prefere 1 a 3 petits paragraphes ou quelques questions courtes plutot qu'une longue liste rigide.
- Ne repete pas a chaque message les memes avertissements generiques si ce n'est pas necessaire.
- Si un signe de gravite apparait dans le contexte ou dans la question, dis clairement qu'il faut contacter rapidement un veterinaire ou une urgence.
- Ne pose jamais de diagnostic definitif.
- N'invente aucune information absente du contexte.
- Si une information n'est pas disponible, dis-le simplement.
- Ecris en francais clair.
- Ecris en texte brut uniquement, sans markdown, sans asterisques et sans citations entre crochets.

Connaissances a utiliser en priorite:
{context_block}
""".strip()

    messages = [{"role": "system", "content": system_content}]
    messages.extend(_normalize_history(history))
    messages.append({"role": "user", "content": question})
    return messages

def stream_vetia_chat(question, history=None, rag=None, k=DEFAULT_TOP_K):
    if should_ask_clarifying_questions(question, history=history):
        clarification = build_clarification_reply()

        def clarification_iterator():
            yield clarification

        return clarification_iterator(), []

    hits = search(question, k=k, rag=rag)
    messages = build_chat_messages(question=question, hits=hits, history=history)

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL_CORE,
            "messages": messages,
            "stream": True,
            "keep_alive": "15m",
            "options": {
                "temperature": 0.2,
                "num_predict": DEFAULT_NUM_PREDICT,
            },
        },
        timeout=180, stream=True
    )
    r.raise_for_status()

    def iterator():
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode())
            message = data.get("message") or {}
            content = message.get("content")
            if content:
                yield content

    return iterator(), [
        {"rag": h["rag"], "path": h["path"], "score": round(h["score"], 4)}
        for h in hits
    ]

def ask_vetia_chat(question, history=None, rag=None, k=DEFAULT_TOP_K):
    chunks, sources = stream_vetia_chat(question=question, history=history, rag=rag, k=k)
    full = "".join(chunks)
    return {
        "answer": sanitize_answer_text(full),
        "sources": sources,
    }

def ask_vetia(question, rag=None, k=DEFAULT_TOP_K):
    return ask_vetia_chat(question, history=None, rag=rag, k=k)["answer"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Question-reponse sur un ou plusieurs RAG")
    parser.add_argument("--rag", type=str, default=None, help="Nom du RAG cible (sinon recherche globale)")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Nombre d'extraits recuperes")
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