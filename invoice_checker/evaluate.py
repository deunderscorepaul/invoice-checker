"""Evaluation harness: per-fault precision / recall / F1 plus latency accounting.

Design decisions that matter for credibility:

* Metrics are computed **per fault type** and macro-averaged. A single aggregate accuracy figure is
  meaningless when ~60% of documents are faulty by construction.
* The prediction space and the ground-truth space are both restricted to injectable codes, so a
  detector cannot inflate its score by reporting a fault that was never injected.
* ``EXTRACTION_FAILED`` and other non-injectable codes are reported separately; hiding them inside
  accuracy is the most common way these projects lie to themselves.
* Document-level metrics are included alongside fault-level ones, because "did we notice anything
  wrong with this invoice" is what a finance team actually cares about.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import OUT_DIR, dump_json, read_jsonl
from .taxonomy import FAULT_BY_CODE, INJECTABLE_CODES


@dataclass
class CodeMetrics:
    code: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    injected: int = 0

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        spec = FAULT_BY_CODE.get(self.code)
        return {
            "code": self.code,
            "severity": spec.severity.value if spec else "info",
            "detector": spec.detector.value if spec else "rules",
            "injected": self.injected,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class EvalReport:
    tag: str
    documents: int = 0
    faulty_documents: int = 0
    clean_documents: int = 0
    codes: dict[str, CodeMetrics] = field(default_factory=dict)
    doc_tp: int = 0
    doc_fp: int = 0
    doc_fn: int = 0
    doc_tn: int = 0
    partial_documents: int = 0
    non_injectable: dict[str, int] = field(default_factory=dict)
    latency_mean_ms: float = 0.0
    latency_p95_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def metric(self, code: str) -> CodeMetrics:
        return self.codes.setdefault(code, CodeMetrics(code=code))

    @property
    def micro(self) -> dict:
        tp = sum(m.tp for m in self.codes.values())
        fp = sum(m.fp for m in self.codes.values())
        fn = sum(m.fn for m in self.codes.values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4),
                "recall": round(recall, 4), "f1": round(f1, 4)}

    @property
    def macro(self) -> dict:
        scored = [m for m in self.codes.values() if m.injected or m.fp]
        if not scored:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "codes_scored": 0}
        return {
            "precision": round(sum(m.precision for m in scored) / len(scored), 4),
            "recall": round(sum(m.recall for m in scored) / len(scored), 4),
            "f1": round(sum(m.f1 for m in scored) / len(scored), 4),
            "codes_scored": len(scored),
        }

    @property
    def document_level(self) -> dict:
        total = self.doc_tp + self.doc_fp + self.doc_fn + self.doc_tn
        accuracy = (self.doc_tp + self.doc_tn) / total if total else 0.0
        precision = self.doc_tp / (self.doc_tp + self.doc_fp) if (self.doc_tp + self.doc_fp) else 0.0
        recall = self.doc_tp / (self.doc_tp + self.doc_fn) if (self.doc_tp + self.doc_fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "tp": self.doc_tp, "fp": self.doc_fp, "fn": self.doc_fn, "tn": self.doc_tn,
            "accuracy": round(accuracy, 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "partial_matches": self.partial_documents,
        }

    def to_dict(self) -> dict:
        by_detector: dict[str, dict] = {}
        for metrics in self.codes.values():
            spec = FAULT_BY_CODE.get(metrics.code)
            kind = spec.detector.value if spec else "rules"
            bucket = by_detector.setdefault(kind, {"codes": 0, "tp": 0, "fp": 0, "fn": 0})
            bucket["codes"] += 1
            bucket["tp"] += metrics.tp
            bucket["fp"] += metrics.fp
            bucket["fn"] += metrics.fn

        return {
            "tag": self.tag,
            "documents": self.documents,
            "faulty_documents": self.faulty_documents,
            "clean_documents": self.clean_documents,
            "micro": self.micro,
            "macro": self.macro,
            "document_level": self.document_level,
            "by_detector": by_detector,
            "per_code": {code: metrics.to_dict() for code, metrics in sorted(self.codes.items())},
            "non_injectable_codes": self.non_injectable,
            "latency_mean_ms": round(self.latency_mean_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
            "notes": self.notes,
        }


def evaluate(predictions_path: Path, tag: str | None = None) -> EvalReport:
    records = list(read_jsonl(predictions_path))
    report = EvalReport(tag=tag or Path(predictions_path).stem)
    injectable = set(INJECTABLE_CODES)
    latencies: list[float] = []

    for record in records:
        report.documents += 1
        truth = {code for code in record.get("truth_codes", []) if code in injectable}
        predicted = {code for code in record.get("codes", []) if code in injectable}
        extra = [code for code in record.get("codes", []) if code not in injectable]
        for code in extra:
            report.non_injectable[code] = report.non_injectable.get(code, 0) + 1

        if truth:
            report.faulty_documents += 1
        else:
            report.clean_documents += 1

        for code in truth:
            report.metric(code).injected += 1
        for code in truth & predicted:
            report.metric(code).tp += 1
        for code in predicted - truth:
            report.metric(code).fp += 1
        for code in truth - predicted:
            report.metric(code).fn += 1

        if truth and predicted:
            if truth == predicted:
                report.doc_tp += 1
            else:
                report.partial_documents += 1
                report.doc_tp += 1          # noticed something, but not everything
        elif truth and not predicted:
            report.doc_fn += 1
        elif predicted and not truth:
            report.doc_fp += 1
        else:
            report.doc_tn += 1

        latency = record.get("latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))

    if latencies:
        latencies.sort()
        report.latency_mean_ms = sum(latencies) / len(latencies)
        index = min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))
        report.latency_p95_ms = latencies[index]

    if report.non_injectable:
        report.notes.append(
            "non-injectable codes reported separately: "
            + ", ".join(f"{code}x{count}" for code, count in sorted(report.non_injectable.items()))
        )
    return report


def render_console(report: EvalReport) -> str:
    lines: list[str] = []
    lines.append(f"=== evaluation: {report.tag} ===")
    lines.append(f"documents            : {report.documents} "
                 f"({report.faulty_documents} faulty / {report.clean_documents} clean)")
    micro = report.micro
    macro = report.macro
    doc = report.document_level
    lines.append(f"fault-level micro    : P={micro['precision']:.3f} "
                 f"R={micro['recall']:.3f} F1={micro['f1']:.3f} "
                 f"(tp={micro['tp']} fp={micro['fp']} fn={micro['fn']})")
    lines.append(f"fault-level macro    : P={macro['precision']:.3f} "
                 f"R={macro['recall']:.3f} F1={macro['f1']:.3f} "
                 f"over {macro['codes_scored']} codes")
    lines.append(f"document-level       : acc={doc['accuracy']:.3f} P={doc['precision']:.3f} "
                 f"R={doc['recall']:.3f} F1={doc['f1']:.3f} "
                 f"(partial={doc['partial_matches']})")
    lines.append(f"latency              : mean={report.latency_mean_ms:.0f} ms "
                 f"p95={report.latency_p95_ms:.0f} ms")
    lines.append("")
    lines.append(f"{'code':<38}{'inj':>4}{'tp':>5}{'fp':>5}{'fn':>5}{'P':>8}{'R':>8}{'F1':>8}")
    lines.append("-" * 81)
    for code, metrics in sorted(report.codes.items(), key=lambda item: (-item[1].injected, item[0])):
        lines.append(
            f"{code:<38}{metrics.injected:>4}{metrics.tp:>5}{metrics.fp:>5}{metrics.fn:>5}"
            f"{metrics.precision:>8.3f}{metrics.recall:>8.3f}{metrics.f1:>8.3f}"
        )
    if report.notes:
        lines.append("")
        for note in report.notes:
            lines.append(f"note: {note}")
    return "\n".join(lines)


def compare_reports(reports: list[EvalReport]) -> str:
    """Side-by-side comparison -- the table that goes straight into the write-up."""
    lines = ["=== comparison ==="]
    header = f"{'variant':<22}{'micro F1':>10}{'macro F1':>10}{'doc F1':>10}{'latency ms':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for report in reports:
        lines.append(
            f"{report.tag:<22}{report.micro['f1']:>10.3f}{report.macro['f1']:>10.3f}"
            f"{report.document_level['f1']:>10.3f}{report.latency_mean_ms:>12.0f}"
        )
    return "\n".join(lines)


def save(report: EvalReport, path: Path | None = None) -> Path:
    path = path or OUT_DIR / f"eval_{report.tag}.json"
    dump_json(path, report.to_dict())
    return path


def load_report(path: Path) -> EvalReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    report = EvalReport(tag=payload["tag"], documents=payload["documents"],
                        faulty_documents=payload["faulty_documents"],
                        clean_documents=payload["clean_documents"])
    for code, metrics in payload["per_code"].items():
        report.codes[code] = CodeMetrics(
            code=code, tp=metrics["tp"], fp=metrics["fp"], fn=metrics["fn"],
            injected=metrics["injected"],
        )
    doc = payload["document_level"]
    report.doc_tp = doc["tp"]
    report.doc_fp = doc["fp"]
    report.doc_fn = doc["fn"]
    report.doc_tn = doc["tn"]
    report.partial_documents = doc.get("partial_matches", 0)
    report.latency_mean_ms = payload.get("latency_mean_ms", 0.0)
    report.latency_p95_ms = payload.get("latency_p95_ms", 0.0)
    report.non_injectable = payload.get("non_injectable_codes", {})
    report.notes = payload.get("notes", [])
    return report
