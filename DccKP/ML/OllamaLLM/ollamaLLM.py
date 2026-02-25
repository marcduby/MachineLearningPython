

# imports
import os
import requests
from typing import Any, Dict, Optional, Tuple
from flask import Flask, request, jsonify
import json


app = Flask(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")

PROMPT = '''
create only one label to summaize this group, docusing on biological process: WP_STATIN_INHIBITION_OF_CHOLESTEROL_PRODUCTION;GOBP_REGULATION_OF_PLASMA_LIPOPROTEIN_PARTICLE_LEVELS;GOBP_PROTEIN_CONTAINING_COMPLEX_REMODELING;WP_METABOLIC_PATHWAY_OF_LDL_HDL_AND_TG_INCLUDING_DISEASES;REACTOME_PLASMA_LIPOPROTEIN_ASSEMBLY_REMODELING_AND_CLEARANCE
'''

TEXT_PATHWAYS = "WP_STATIN_INHIBITION_OF_CHOLESTEROL_PRODUCTION;GOBP_REGULATION_OF_PLASMA_LIPOPROTEIN_PARTICLE_LEVELS;GOBP_PROTEIN_CONTAINING_COMPLEX_REMODELING;WP_METABOLIC_PATHWAY_OF_LDL_HDL_AND_TG_INCLUDING_DISEASES;REACTOME_PLASMA_LIPOPROTEIN_ASSEMBLY_REMODELING_AND_CLEARANCE"
MODEL_GEMMA2 = "gemma2:2b"

LIST_PATHWAY_FACTORS = [
    "GOBP_REGULATION_OF_LIPID_LOCALIZATION;GOBP_REGULATION_OF_LIPID_TRANSPORT;GOBP_REGULATION_OF_STEROL_TRANSPORT;GOBP_POSITIVE_REGULATION_OF_LIPID_LOCALIZATION;GOBP_CHOLESTEROL_EFFLUX",
    "WP_STATIN_INHIBITION_OF_CHOLESTEROL_PRODUCTION;GOBP_REGULATION_OF_PLASMA_LIPOPROTEIN_PARTICLE_LEVELS;GOBP_PROTEIN_CONTAINING_COMPLEX_REMODELING;WP_METABOLIC_PATHWAY_OF_LDL_HDL_AND_TG_INCLUDING_DISEASES;REACTOME_PLASMA_LIPOPROTEIN_ASSEMBLY_REMODELING_AND_CLEARANCE",
    "GOBP_RESPONSE_TO_OXYGEN_CONTAINING_COMPOUND;GOBP_REGULATION_OF_RESPONSE_TO_EXTERNAL_STIMULUS;GOBP_CELLULAR_RESPONSE_TO_OXYGEN_CONTAINING_COMPOUND;GOBP_CIRCULATORY_SYSTEM_PROCESS;GOBP_RESPONSE_TO_ENDOGENOUS_STIMULUS",
    "GOBP_BLOOD_VESSEL_MORPHOGENESIS;GOBP_VASCULATURE_DEVELOPMENT;GOBP_TUBE_MORPHOGENESIS;GOBP_TUBE_DEVELOPMENT;GOBP_CIRCULATORY_SYSTEM_DEVELOPMENT"
]

class OllamaError(Exception):
    """Raised for Ollama-related errors with a user-friendly message."""


def get_ollama_generate_summary(
    text: str,
    model: str = DEFAULT_MODEL,
    *,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout_s: Tuple[float, float] = (3.0, 60.0),  # (connect timeout, read timeout)
) -> Dict[str, Any]:
    """
    Generate a summary using Ollama's /api/generate.

    Returns a dict with fields:
      - ok: bool
      - summary: str (if ok)
      - error: str (if not ok)
      - details: any optional debug details
    """

    if not isinstance(text, str) or not text.strip():
        return {"ok": False, "error": "Input 'text' must be a non-empty string."}

    # Prompt for a small model: keep it short and explicit.
    # prompt = (
    #     "Summarize the following text in 3-6 bullet points. "
    #     "Be concise and preserve key facts.\n\n"
    #     f"TEXT:\n{text.strip()}\n"
    # )

    prompt = (
        "create only one label to summaize this group of gene pathways, focusing on biological process: "
        "Be concise and preserve key facts.\n\n"
        f"TEXT:\n{text.strip()}\n"
    )

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
    }
    if max_tokens is not None:
        # Ollama options often accept num_predict (token prediction limit).
        payload["options"]["num_predict"] = int(max_tokens)

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"

    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "error": "Cannot connect to Ollama. Is it running? Try: `ollama serve`.",
            "details": {"url": url},
        }
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "error": "Ollama request timed out.",
            "details": {"timeout_s": timeout_s},
        }
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"Request to Ollama failed: {e.__class__.__name__}"}

    # Non-200 responses: try to extract useful error details.
    if resp.status_code != 200:
        err_text = resp.text[:1000] if resp.text else ""
        # Ollama often returns JSON error bodies; parse if possible.
        try:
            err_json = resp.json()
        except ValueError:
            err_json = None

        return {
            "ok": False,
            "error": f"Ollama returned HTTP {resp.status_code}.",
            "details": {"body": err_json if err_json is not None else err_text},
        }

    # Parse JSON success payload
    try:
        data = resp.json()
    except ValueError:
        return {"ok": False, "error": "Ollama returned invalid JSON."}

    # Ollama /api/generate returns fields like: response, done, etc.
    summary = data.get("response")
    if not isinstance(summary, str) or not summary.strip():
        return {
            "ok": False,
            "error": "Ollama returned an empty response.",
            "details": {"data_keys": list(data.keys())},
        }

    return {"ok": True, "summary": summary.strip(), "details": {"model": model}}



if __name__ == "__main__":

    for factor in LIST_PATHWAY_FACTORS:
        result = get_ollama_generate_summary(text=factor, model=MODEL_GEMMA2)

        # log
        print("factor: {}".format(factor))
        print("result: \n{}\n\n".format(json.dumps(result, indent=2)))