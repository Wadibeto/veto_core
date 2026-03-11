import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import requests

from query import ask_vetia_chat, stream_vetia_chat, MODEL_CORE, OLLAMA_URL

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
INDEX_HTML = PROJECT_ROOT / "index2.html"
MODELFILE = PROJECT_ROOT / "Modelfile"
HOST = os.getenv("VETO_API_HOST", "127.0.0.1")
PORT = int(os.getenv("VETO_API_PORT", "8000"))


def _is_ollama_running():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def _start_ollama_background():
    kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }

    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    subprocess.Popen(["ollama", "serve"], **kwargs)


def _wait_for_ollama(timeout_seconds=20):
    started_at = time.time()
    while time.time() - started_at < timeout_seconds:
        if _is_ollama_running():
            return True
        time.sleep(0.5)
    return False


def _ensure_ollama_running():
    if _is_ollama_running():
        return

    try:
        _start_ollama_background()
    except FileNotFoundError as ex:
        raise RuntimeError("Ollama n'est pas installe ou n'est pas dans le PATH.") from ex

    if not _wait_for_ollama():
        raise RuntimeError("Impossible de demarrer Ollama automatiquement.")


def _ensure_model_exists():
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    response.raise_for_status()
    models = response.json().get("models", [])

    for model in models:
        name = model.get("name", "")
        model_base = model.get("model", "")
        if name == MODEL_CORE or name.startswith(f"{MODEL_CORE}:"):
            return
        if model_base == MODEL_CORE or model_base.startswith(f"{MODEL_CORE}:"):
            return

    if not MODELFILE.exists():
        raise RuntimeError(f"Modelfile introuvable: {MODELFILE}")

    subprocess.run(["ollama", "create", MODEL_CORE, "-f", str(MODELFILE)], check=True)


def ensure_runtime_ready():
    _ensure_ollama_running()
    _ensure_model_exists()


def _json_response(handler, status, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


class ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep server logs concise in terminal.
        return

    def _serve_index(self):
        if not INDEX_HTML.exists():
            self.send_error(404, "index2.html introuvable")
            return

        data = INDEX_HTML.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index2.html"):
            self._serve_index()
            return

        if parsed.path == "/api/health":
            _json_response(self, 200, {"ok": True})
            return

        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/api/chat", "/api/chat/stream"):
            self.send_error(404, "Not found")
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            _json_response(self, 400, {"error": "JSON invalide"})
            return

        message = str(payload.get("message", "")).strip()
        history = payload.get("history") or []
        rag = payload.get("rag") or None
        k = payload.get("k", 6)

        if not message:
            _json_response(self, 400, {"error": "Le champ 'message' est requis."})
            return

        try:
            k = int(k)
        except Exception:
            k = 6

        if parsed.path == "/api/chat/stream":
            try:
                chunks, sources = stream_vetia_chat(question=message, history=history, rag=rag, k=k)
            except Exception as ex:
                _json_response(self, 500, {"error": f"Erreur backend: {str(ex)}"})
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            answer_parts = []
            try:
                for chunk in chunks:
                    answer_parts.append(chunk)
                    line = json.dumps({"type": "chunk", "content": chunk}, ensure_ascii=False) + "\n"
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()

                done_line = json.dumps({"type": "done", "sources": sources}, ensure_ascii=False) + "\n"
                self.wfile.write(done_line.encode("utf-8"))
                self.wfile.flush()
            except BrokenPipeError:
                return

            return

        try:
            result = ask_vetia_chat(question=message, history=history, rag=rag, k=k)
        except Exception as ex:
            _json_response(self, 500, {"error": f"Erreur backend: {str(ex)}"})
            return

        _json_response(self, 200, result)


if __name__ == "__main__":
    ensure_runtime_ready()
    server = ThreadingHTTPServer((HOST, PORT), ChatHandler)
    print(f"API VetIA active sur http://{HOST}:{PORT}")
    print("Routes: GET /, GET /api/health, POST /api/chat, POST /api/chat/stream")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
