"""Kleiner Ollama-Client und JSON-Schema-Prüfer, nur Standardbibliothek.

Das Modell wird per Grammatik ins Schema gezwungen. Damit muss die Pipeline keinen
Fließtext, keine Markdown-Zäune und keine Kommas am Ende wegputzen -- das macht
auch ein kleines Modell für die Extraktion brauchbar.

Der Prüfer sagt genau, was nicht passt; die Meldung geht beim Wiederholungsversuch
an das Modell zurück.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

DEFAULT_TIMEOUT = 180


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """Raised when no Ollama server is reachable -- the caller should degrade gracefully."""


class SchemaError(LLMError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


# --- JSON schema validation ---------------------------------------------------------------------

def validate_against_schema(instance, schema: dict, path: str = "$") -> list[str]:
    """Return a list of human-readable violations. Empty list means valid.

    Supports a union ``type`` list (for example ``["string", "null"]``), which the invoice schema
    uses so the model can explicitly report "this field is not present" instead of inventing a value.
    """
    errors: list[str] = []
    expected_type = schema.get("type")

    if isinstance(expected_type, list):
        # Union type: accept the value if it satisfies any member. Null is always acceptable when
        # listed, and is checked before recursing so subschemas never see a None.
        if instance is None:
            return [] if "null" in expected_type else [f"{path}: null is not permitted"]
        member_errors: list[list[str]] = []
        for member in expected_type:
            if member == "null":
                continue
            candidate_errors = validate_against_schema(instance, {**schema, "type": member}, path)
            if not candidate_errors:
                return []
            member_errors.append(candidate_errors)
        return member_errors[0] if member_errors else [f"{path}: no permitted type matched"]

    # A null for a non-nullable field is a violation, not a type mismatch to explain at length.
    if instance is None:
        return [f"{path}: required field is null"]

    if expected_type == "object":
        if not isinstance(instance, dict):
            return [f"{path}: expected object, got {type(instance).__name__}"]
        for key in schema.get("required", []):
            if key not in instance or instance[key] is None:
                errors.append(f"{path}.{key}: required field missing")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in instance:
                if key not in properties:
                    errors.append(f"{path}.{key}: unexpected field")
        for key, subschema in properties.items():
            if key in instance and instance[key] is not None:
                errors.extend(validate_against_schema(instance[key], subschema, f"{path}.{key}"))
        return errors

    if expected_type == "array":
        if not isinstance(instance, list):
            return [f"{path}: expected array, got {type(instance).__name__}"]
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(instance):
                errors.extend(validate_against_schema(item, item_schema, f"{path}[{index}]"))
        return errors

    if expected_type == "string":
        if not isinstance(instance, str):
            errors.append(f"{path}: expected string, got {type(instance).__name__}")
        elif "enum" in schema and instance not in schema["enum"]:
            errors.append(f"{path}: '{instance}' is not one of {schema['enum']}")
        elif schema.get("format") == "date" and instance.strip():
            # An empty string means "this field is absent from the document" and must pass, even for
            # a date-typed field: many valid invoices carry no service date, and rejecting them would
            # trigger a pointless repair round-trip on every such document.
            from datetime import datetime

            try:
                datetime.strptime(instance, "%Y-%m-%d")
            except ValueError:
                errors.append(f"{path}: '{instance}' is not a YYYY-MM-DD date")
        return errors

    if expected_type in {"number", "integer"}:
        if isinstance(instance, bool) or not isinstance(instance, (int, float)):
            errors.append(f"{path}: expected {expected_type}, got {type(instance).__name__}")
        elif expected_type == "integer" and float(instance) != int(instance):
            errors.append(f"{path}: expected integer, got {instance}")
        return errors

    if expected_type == "boolean":
        if not isinstance(instance, bool):
            errors.append(f"{path}: expected boolean, got {type(instance).__name__}")
        return errors

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: '{instance}' is not one of {schema['enum']}")
    return errors


def parse_json_object(raw: str) -> dict:
    """Best-effort recovery of a JSON object from model output.

    With grammar-constrained decoding this is a formality; it exists so an unconstrained or
    locally-run model (llama.cpp without a grammar, a fine-tuned checkpoint) still has a path.
    """
    text = (raw or "").strip()
    if not text:
        raise SchemaError(["$: model returned an empty response"])
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise SchemaError(["$: response contained no JSON object"])
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise SchemaError([f"$: invalid JSON ({exc.msg} at position {exc.pos})"]) from exc
    if not isinstance(payload, dict):
        raise SchemaError([f"$: expected a JSON object, got {type(payload).__name__}"])
    return payload


# --- Ollama transport ---------------------------------------------------------------------------

# --- Ollama transport ---------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434",
                 timeout: int = DEFAULT_TIMEOUT, temperature: float = 0.0,
                 num_ctx: int = 4096):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_ms": 0.0}
        self.last_timing: dict = {}

    def _request(self, path: str, payload: dict | None, method: str = "POST"):
        url = f"{self.host}{path}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, method=method,
                                        headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMUnavailable(
                f"could not reach Ollama at {self.host} ({exc.reason}). "
                f"Start it with 'ollama serve' and pull the model ('ollama pull {self.model}'), "
                f"or run with --baseline to use the rule engine alone."
            ) from exc
        except urllib.error.HTTPError as exc:
            raise LLMError(f"Ollama returned HTTP {exc.code}: {exc.read()[:300]!r}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"Ollama returned a non-JSON response: {exc}") from exc

    def _request_stream(self, path: str, payload: dict, on_token=None):
        """Ollama-Stream (newline-delimited JSON) lesen und Stücke weitergeben.

        Ohne Streaming kommt erst am Ende Ausgabe. Eine langsame Generierung ist
        dann nicht von einer hängenden zu unterscheiden.
        """
        url = f"{self.host}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST",
                                        headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        raise LLMError(f"Ollama reported: {chunk['error']}")
                    piece = chunk.get("response", "")
                    if piece and on_token is not None:
                        on_token(piece, chunk)
                    yield chunk
        except urllib.error.URLError as exc:
            raise LLMUnavailable(f"could not reach Ollama at {self.host} ({exc.reason})") from exc
        except urllib.error.HTTPError as exc:
            raise LLMError(f"Ollama returned HTTP {exc.code}: {exc.read()[:300]!r}") from exc

    def list_models(self) -> list[str]:
        payload = self._request("/api/tags", None, method="GET")
        return [entry.get("name", "") for entry in payload.get("models", [])]

    def loaded_models(self) -> list[dict]:
        """Currently resident models, with the GPU/CPU memory split that explains slow generation."""
        payload = self._request("/api/ps", None, method="GET")
        return payload.get("models", []) or []

    def residency(self) -> dict:
        """Report whether the model is fully on the GPU.

        This is the diagnostic for "it says 4 minutes remaining and never changes": a model that does
        not fit in VRAM is partially offloaded to the CPU, and the CPU-side layers dominate the
        runtime, often by an order of magnitude.
        """
        for entry in self.loaded_models():
            if entry.get("name", "").split(":")[0] != self.model.split(":")[0]:
                continue
            total = float(entry.get("size") or 0)
            vram = float(entry.get("size_vram") or 0)
            on_gpu = (vram / total) if total else 0.0
            return {
                "loaded": True,
                "name": entry.get("name"),
                "size_gb": round(total / 1e9, 2),
                "vram_gb": round(vram / 1e9, 2),
                "gpu_fraction": round(on_gpu, 3),
                "fully_on_gpu": bool(total and vram >= total * 0.995),
                "context_length": entry.get("context_length"),
            }
        return {"loaded": False, "name": self.model}

    def health(self, *, timeout: float = 10.0) -> tuple[bool, str]:
        try:
            models = self.list_models()
        except LLMError as exc:
            return False, str(exc)
        if not any(name.split(":")[0] == self.model.split(":")[0] for name in models):
            return False, (
                f"model '{self.model}' not found locally; available: {', '.join(models) or 'none'}. "
                f"Run: ollama pull {self.model}"
            )
        return True, f"Ollama reachable, model '{self.model}' present"
        if not any(name.split(":")[0] == self.model.split(":")[0] for name in models):
            return False, (
                f"model '{self.model}' not found locally; available: {', '.join(models) or 'none'}. "
                f"Run: ollama pull {self.model}"
            )
        return True, f"Ollama reachable, model '{self.model}' present"

    def generate_json(self, prompt: str, system: str, schema: dict, *,
                      max_retries: int = 1, on_progress=None,
                      progress_interval: float = 1.0) -> tuple[dict, str]:
        """One constrained call with at most ``max_retries`` repair attempts, streamed.

        Returns ``(payload, raw_response)``. Raises :class:`SchemaError` when the schema cannot be
        satisfied, which the caller reports as an extraction failure rather than guessing.

        ``on_progress`` receives a dict roughly every ``progress_interval`` seconds with the token
        count and instantaneous speed, so a slow generation is visibly slow rather than apparently
        hung. Streaming also gives real timings from Ollama, which makes the "is this running on the
        GPU?" question answerable from the run itself.
        """
        attempt_prompt = prompt
        last_raw = ""
        last_errors: list[str] = []
        for attempt in range(max_retries + 1):
            started = time.perf_counter()
            body_parts: list[str] = []
            final_chunk: dict = {}
            token_count = 0
            last_report = started

            def handle_token(piece: str, chunk: dict) -> None:
                nonlocal token_count, last_report
                body_parts.append(piece)
                token_count += 1
                now = time.perf_counter()
                if on_progress is not None and now - last_report >= progress_interval:
                    last_report = now
                    elapsed = now - started
                    on_progress({
                        "attempt": attempt,
                        "tokens": token_count,
                        "elapsed_s": elapsed,
                        "tokens_per_s": token_count / elapsed if elapsed > 0 else 0.0,
                    })

            for chunk in self._request_stream("/api/generate", {
                "model": self.model,
                "prompt": attempt_prompt,
                "system": system,
                "stream": True,
                "format": schema,
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                    "seed": 0,
                },
            }, on_token=handle_token):
                final_chunk = chunk

            elapsed_ms = (time.perf_counter() - started) * 1000
            self.stats["calls"] += 1
            self.stats["total_ms"] += elapsed_ms
            self.stats["prompt_tokens"] += final_chunk.get("prompt_eval_count", 0) or 0
            self.stats["completion_tokens"] += final_chunk.get("eval_count", 0) or 0

            # Ollama reports durations in nanoseconds; convert to a per-token speed for the report.
            eval_count = final_chunk.get("eval_count") or 0
            eval_ns = final_chunk.get("eval_duration") or 0
            prompt_ns = final_chunk.get("prompt_eval_duration") or 0
            self.last_timing = {
                "elapsed_ms": round(elapsed_ms, 1),
                "prompt_tokens": final_chunk.get("prompt_eval_count") or 0,
                "completion_tokens": eval_count,
                "generate_tokens_per_s": round(eval_count / (eval_ns / 1e9), 1) if eval_ns else None,
                "prompt_eval_ms": round(prompt_ns / 1e6, 1) if prompt_ns else None,
                "total_duration_ms": round((final_chunk.get("total_duration") or 0) / 1e6, 1),
            }

            last_raw = final_chunk.get("response") or "".join(body_parts)
            try:
                candidate = parse_json_object(last_raw)
            except SchemaError as exc:
                last_errors = exc.errors
                candidate = None
            if candidate is not None:
                last_errors = validate_against_schema(candidate, schema)
                if not last_errors:
                    return candidate, last_raw

            attempt_prompt = (
                f"{prompt}\n\nYour previous answer was rejected by the schema validator.\n"
                f"Violations:\n- " + "\n- ".join(last_errors) +
                "\nReturn a corrected JSON object that satisfies every constraint exactly."
            )
        raise SchemaError(last_errors or ["$: model failed to produce schema-valid JSON"])

    def summarise_stats(self) -> dict:
        calls = max(1, self.stats["calls"])
        return {
            "calls": self.stats["calls"],
            "prompt_tokens": self.stats["prompt_tokens"],
            "completion_tokens": self.stats["completion_tokens"],
            "avg_ms": round(self.stats["total_ms"] / calls, 1),
        }
