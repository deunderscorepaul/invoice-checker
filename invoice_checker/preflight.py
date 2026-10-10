"""Preflight diagnostics: is the model actually running on the GPU?

A model that does not fit in VRAM is partially executed on the CPU, and the CPU-side layers dominate
wall-clock time — often by more than an order of magnitude. Ollama does not say this loudly; it
reports a completion estimate that creeps or stalls. This module turns that into an explicit
measurement before a long run is started.
"""
from __future__ import annotations

import time

from .llm import LLMError, OllamaClient

# Rough rule of thumb for Q4_K_M weights plus context and compute buffers.
MIN_HEADROOM_GB = 1.5


def describe_model(model: str, client: OllamaClient) -> list[str]:
    """Static facts about the model as recorded by Ollama."""
    lines: list[str] = []
    try:
        payload = client._request("/api/tags", None, method="GET")
    except LLMError as exc:
        return [f"could not read model list: {exc}"]

    entry = next((m for m in payload.get("models", [])
                  if m.get("name", "").split(":")[0] == model.split(":")[0]), None)
    if entry is None:
        return [f"model '{model}' is not installed. Run: ollama pull {model}"]

    details = entry.get("details", {}) or {}
    size_gb = (entry.get("size") or 0) / 1e9
    lines.append(f"model            : {entry.get('name')}")
    lines.append(f"parameters       : {details.get('parameter_size', 'unknown')}")
    lines.append(f"quantisation     : {details.get('quantization_level', 'unknown')}")
    lines.append(f"weights on disk  : {size_gb:.2f} GB")
    return lines


def probe(client: OllamaClient, *, timeout_seconds: int = 120) -> dict:
    """Run one tiny generation and report speed plus GPU residency.

    Returns a dict with the observed tokens/second and whether the model fit in VRAM, plus advice.
    """
    started = time.perf_counter()
    result = {
        "ok": False,
        "tokens_per_s": None,
        "elapsed_s": None,
        "residency": {},
        "advice": [],
    }

    try:
        payload, _ = client.generate_json(
            "Document:\nInvoice No. TEST-1\nInvoice date: 2024-01-01\n"
            "Gross total: 119,00 EUR\n\nJSON:",
            "You extract structured fields from invoices. Answer with JSON only.",
            _probe_schema(),
            max_retries=0,
        )
    except LLMError as exc:
        result["error"] = str(exc)
        return result

    elapsed = time.perf_counter() - started
    timing = client.last_timing or {}
    result["elapsed_s"] = round(elapsed, 2)
    result["tokens_per_s"] = timing.get("generate_tokens_per_s")
    result["timing"] = timing
    result["ok"] = True

    try:
        result["residency"] = client.residency()
    except LLMError:
        result["residency"] = {}

    residency = result["residency"]
    speed = result["tokens_per_s"] or 0

    if residency.get("loaded") and not residency.get("fully_on_gpu"):
        result["advice"].append(
            f"ONLY PART OF THE MODEL IS ON THE GPU "
            f"({residency.get('vram_gb')} GB of {residency.get('size_gb')} GB). "
            f"The rest runs on the CPU, which is why progress crawls."
        )
    if speed and speed < 8:
        result["advice"].append(
            f"generation speed is {speed:.1f} tokens/s, which indicates CPU offload or a very small "
            f"context. Expect roughly {(200 / speed) if speed else 0:.0f}s per invoice."
        )
    if not speed:
        result["advice"].append("Ollama did not report a generation speed; the model may be cold.")
    return result


def _probe_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["invoice_number", "gross_amount"],
        "properties": {
            "invoice_number": {"type": "string"},
            "gross_amount": {"type": "number"},
        },
    }


def format_report(model: str, client: OllamaClient, result: dict) -> str:
    lines = ["=== preflight ===", f"host             : {client.host}"]
    lines.extend(describe_model(model, client))
    lines.append(f"context (num_ctx) : {client.num_ctx}")

    if not result.get("ok"):
        lines.append("")
        lines.append(f"probe FAILED: {result.get('error', 'unknown error')}")
        return "\n".join(lines)

    timing = result.get("timing", {})
    lines.append("")
    lines.append(f"probe elapsed    : {result['elapsed_s']} s")
    lines.append(f"prompt tokens    : {timing.get('prompt_tokens')}")
    lines.append(f"output tokens    : {timing.get('completion_tokens')}")
    lines.append(f"generation speed : {result['tokens_per_s']} tokens/s")
    lines.append(f"prompt eval      : {timing.get('prompt_eval_ms')} ms")

    residency = result.get("residency") or {}
    lines.append("")
    if residency.get("loaded"):
        verdict = "fully on GPU" if residency.get("fully_on_gpu") else "SPLIT ACROSS GPU AND CPU"
        lines.append(f"residency        : {verdict}")
        lines.append(f"  weights in VRAM: {residency.get('vram_gb')} GB "
                     f"of {residency.get('size_gb')} GB")
    else:
        lines.append("residency        : model not resident after the probe "
                     "(it may have been unloaded immediately)")

    if result.get("advice"):
        lines.append("")
        for note in result["advice"]:
            lines.append(f"ADVICE: {note}")
    return "\n".join(lines)
