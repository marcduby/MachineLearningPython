


#!/usr/bin/env python3
"""
Flask + Ollama Web API (GET /ask?prompt=...).

Security features included:
- API key required (X-API-Key header; query param fallback optional)
- Basic in-memory rate limiting (per IP)
- Optional IP allowlist (ALLOWED_IPS env var, supports CIDR)
- Input size/character checks + prompt length cap
- Timeouts + sane error handling
- Minimal logging (doesn't log prompt content)

Ollama API:
- Uses POST {OLLAMA_BASE_URL}/api/chat
"""

import os
import re
import time
import json
import ipaddress
from collections import deque

import requests
from flask import Flask, request, jsonify, abort

app = Flask(__name__)

# ---------------------------
# Config (env overridable)
# ---------------------------
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a computational biologist. Be precise, cite assumptions, and keep answers concise."
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_TIMEOUT_SECS = float(os.getenv("OLLAMA_TIMEOUT_SECS", "60"))

# API key (required)
API_KEY = os.getenv("API_KEY", "")
if not API_KEY:
    raise RuntimeError("Missing API_KEY environment variable. Refusing to start.")

# Request limits
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "4000"))
MAX_MODEL_CHARS = int(os.getenv("MAX_MODEL_CHARS", "64"))

# Rate limiting: N requests per window seconds per IP
RATE_LIMIT_N = int(os.getenv("RATE_LIMIT_N", "30"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# Optional IP allowlist: comma-separated (supports CIDR)
ALLOWED_IPS = [s.strip() for s in os.getenv("ALLOWED_IPS", "").split(",") if s.strip()]

# Optional: allow API key via query param (less safe; defaults off)
ALLOW_QUERY_KEY = os.getenv("ALLOW_QUERY_KEY", "0") == "1"

# ---------------------------
# Very small in-memory limiter
# ---------------------------
_requests = {}  # ip -> deque[timestamps]


def get_client_ip() -> str:
    """
    If you are behind a reverse proxy, do NOT blindly trust X-Forwarded-For
    unless you've configured ProxyFix / trusted proxy settings.
    """
    return request.remote_addr or "unknown"


def ip_allowed(ip_str: str) -> bool:
    if not ALLOWED_IPS:
        return True
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    for entry in ALLOWED_IPS:
        try:
            if "/" in entry:
                net = ipaddress.ip_network(entry, strict=False)
                if ip_obj in net:
                    return True
            else:
                if ip_obj == ipaddress.ip_address(entry):
                    return True
        except ValueError:
            continue
    return False


def rate_limit_ok(ip: str) -> bool:
    now = time.time()
    dq = _requests.setdefault(ip, deque())
    cutoff = now - RATE_LIMIT_WINDOW
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_N:
        return False
    dq.append(now)
    return True


def require_api_key():
    supplied = request.headers.get("X-API-Key", "")
    if not supplied and ALLOW_QUERY_KEY:
        supplied = request.args.get("key", "")
    if not supplied or supplied != API_KEY:
        abort(401, description="Unauthorized")


def sanitize_model(model: str) -> str:
    model = (model or "").strip()
    if not model:
        return DEFAULT_MODEL
    if len(model) > MAX_MODEL_CHARS:
        abort(400, description="Model name too long")
    # Allow only typical ollama model chars: letters, numbers, . _ : - /
    if not re.fullmatch(r"[A-Za-z0-9._:/-]+", model):
        abort(400, description="Invalid model name")
    return model


def sanitize_prompt(prompt: str) -> str:
    if prompt is None:
        abort(400, description="Missing prompt")
    prompt = prompt.strip()
    if not prompt:
        abort(400, description="Empty prompt")
    if len(prompt) > MAX_PROMPT_CHARS:
        abort(413, description=f"Prompt too large (max {MAX_PROMPT_CHARS} chars)")
    # Reject control chars (except \n, \t)
    if re.search(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", prompt):
        abort(400, description="Prompt contains illegal control characters")
    return prompt


def call_ollama_chat(prompt: str, model: str) -> dict:
    """
    Calls Ollama's /api/chat with system+user messages.
    Uses non-streaming response for simplicity.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # Optional knobs you can expose later:
        # "options": {"temperature": 0.2, "num_ctx": 4096},
    }

    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECS)
    except requests.Timeout:
        abort(504, description="Ollama timed out")
    except requests.RequestException as e:
        abort(502, description=f"Error calling Ollama: {type(e).__name__}")

    if resp.status_code != 200:
        # Keep error short; don't leak internals
        snippet = resp.text[:400]
        abort(502, description=f"Ollama HTTP {resp.status_code}: {snippet}")

    try:
        data = resp.json()
    except ValueError:
        abort(502, description="Ollama returned non-JSON response")

    # Expected shape: {"message": {"role":"assistant","content":"..."}, ...}
    msg = (data.get("message") or {})
    answer = msg.get("content", "")
    if not isinstance(answer, str):
        abort(502, description="Unexpected Ollama response format")

    return {"text": answer, "raw": data}


@app.before_request
def security_gate():
    ip = get_client_ip()
    if not ip_allowed(ip):
        abort(403, description="Forbidden")
    if not rate_limit_ok(ip):
        abort(429, description="Rate limit exceeded")
    require_api_key()


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/ask")
def ask():
    # GET /ask?prompt=...&model=...
    prompt = sanitize_prompt(request.args.get("prompt"))
    model = sanitize_model(request.args.get("model"))

    result = call_ollama_chat(prompt=prompt, model=model)

    return jsonify(
        {
            "model": model,
            "answer": result["text"],
        }
    )


if __name__ == "__main__":
    # Don't expose Flask dev server publicly. Use gunicorn + HTTPS in production.
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8082"))
    app.run(host=host, port=port, debug=False)


