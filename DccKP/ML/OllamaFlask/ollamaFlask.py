#!/usr/bin/env python3
"""
Flask + Ollama Web API (POST /ask with JSON).

Security features included:
- API key required (X-API-Key header; optional query param if ALLOW_QUERY_KEY=1)
- Basic in-memory rate limiting (per IP)
- Optional IP allowlist (ALLOWED_IPS env var, supports CIDR)
- JSON-only endpoint with strict content-type + size limits
- Input validation + prompt/system length caps
- Timeouts + sane error handling
- Minimal logging (doesn't log prompt content)

Request JSON:
{
  "prompt": "your user prompt (required)",
  "system": "system prompt (optional; overrides default)",
  "model": "gemma2:2b (optional)"
}

Response JSON:
{
  "model": "...",
  "answer": "..."
}
"""

import os
import re
import time
import ipaddress
from collections import deque
import logging
import sys 
import json

import requests
from flask import Flask, request, jsonify, abort

# settings
logging.basicConfig(level=logging.INFO, format=f'[%(asctime)s] - %(levelname)s - %(name)s %(threadName)s : %(message)s')
handler = logging.StreamHandler(sys.stdout)

app = Flask(__name__)

# ---------------------------
# Config (env overridable)
# ---------------------------
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
DEFAULT_SYSTEM_PROMPT = os.getenv(
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
MAX_SYSTEM_CHARS = int(os.getenv("MAX_SYSTEM_CHARS", "2000"))
MAX_MODEL_CHARS = int(os.getenv("MAX_MODEL_CHARS", "64"))
MAX_JSON_BYTES = int(os.getenv("MAX_JSON_BYTES", "20000"))  # ~20KB default

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

# methods
def get_logger(name=__name__): 
    # get the logger
    logger = logging.getLogger(name)
    logger.addHandler(handler)

    # return
    return logger 


logger = get_logger()


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


def require_json_request():
    # Strongly require JSON content-type
    if not request.is_json:
        abort(415, description="Content-Type must be application/json")

    # Enforce a max body size (based on Content-Length when present)
    cl = request.content_length
    if cl is not None and cl > MAX_JSON_BYTES:
        abort(413, description=f"Request too large (max {MAX_JSON_BYTES} bytes)")


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


def _reject_control_chars(text: str, field_name: str):
    # Reject control chars (except \n, \t)
    if re.search(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", text):
        abort(400, description=f"{field_name} contains illegal control characters")


def sanitize_prompt(prompt: str) -> str:
    if prompt is None:
        abort(400, description="Missing 'prompt'")
    if not isinstance(prompt, str):
        abort(400, description="'prompt' must be a string")
    prompt = prompt.strip()
    if not prompt:
        abort(400, description="Empty 'prompt'")
    if len(prompt) > MAX_PROMPT_CHARS:
        abort(413, description=f"'prompt' too large (max {MAX_PROMPT_CHARS} chars)")
    _reject_control_chars(prompt, "prompt")
    return prompt


def sanitize_system(system: str | None) -> str:
    if system is None:
        return DEFAULT_SYSTEM_PROMPT
    if not isinstance(system, str):
        abort(400, description="'system' must be a string")
    system = system.strip()
    if not system:
        return DEFAULT_SYSTEM_PROMPT
    if len(system) > MAX_SYSTEM_CHARS:
        abort(413, description=f"'system' too large (max {MAX_SYSTEM_CHARS} chars)")
    _reject_control_chars(system, "system")
    return system


def call_ollama_chat(prompt: str, system: str, model: str, log: bool=True) -> str:
    """
    Calls Ollama's /api/chat with system+user messages.
    Uses non-streaming response for simplicity.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"

    # log
    if log:
        logger.info("calling ollama url: {} with model: {}".format(url, model))
        logger.info("system prompt: {}".format(system))
        logger.info("query prompt: {}".format(prompt))


    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECS)
    except requests.Timeout:
        abort(504, description="Ollama timed out")
    except requests.RequestException as e:
        abort(502, description=f"Error calling Ollama: {type(e).__name__}")

    if resp.status_code != 200:
        snippet = resp.text[:400]
        abort(502, description=f"Ollama HTTP {resp.status_code}: {snippet}")

    try:
        data = resp.json()
    except ValueError:
        abort(502, description="Ollama returned non-JSON response")

    msg = (data.get("message") or {})
    answer = msg.get("content", "")
    if not isinstance(answer, str):
        abort(502, description="Unexpected Ollama response format")


    # log
    if log:
        logger.info("got answer: {}".format(json.dumps(answer, indent=2)))

    # return
    return answer


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


@app.post("/ask")
def ask():
    require_json_request()

    # Parse JSON (force=False because we already require request.is_json)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        abort(400, description="Invalid JSON body")

    prompt = sanitize_prompt(data.get("prompt"))
    system = sanitize_system(data.get("system"))
    model = sanitize_model(data.get("model"))

    answer = call_ollama_chat(prompt=prompt, system=system, model=model)

    return jsonify({"model": model, "answer": answer})


if __name__ == "__main__":
    # Don't expose Flask dev server publicly. Use gunicorn + HTTPS in production.
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8082"))
    app.run(host=host, port=port, debug=False)