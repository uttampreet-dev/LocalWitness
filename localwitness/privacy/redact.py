"""Presidio PII redaction — names, emails, phones, IDs. Fully local. [Phase 2]"""

import time

from localwitness import metrics

# Entity types we redact, and the typed tag each becomes.
PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "CREDIT_CARD",
    "IBAN_CODE",
]

_analyzer = None
_anonymizer = None


def _get_engines():
    global _analyzer, _anonymizer
    if _analyzer is None:
        # Deferred: presidio + spaCy en_core_web_lg take seconds to load and
        # ~600 MB of RAM — only pay that when redaction is actually used.
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        start = time.perf_counter()
        _analyzer = AnalyzerEngine()  # spaCy en_core_web_lg, all local
        _anonymizer = AnonymizerEngine()
        metrics.record_timing("redact_load_s", time.perf_counter() - start)
    return _analyzer, _anonymizer


def redact(text: str) -> str:
    """Replace detected PII with typed tags like [PERSON] or [EMAIL_ADDRESS]."""
    from presidio_anonymizer.entities import OperatorConfig

    analyzer, anonymizer = _get_engines()
    start = time.perf_counter()
    findings = analyzer.analyze(text=text, entities=PII_ENTITIES, language="en")
    operators = {
        entity: OperatorConfig("replace", {"new_value": f"[{entity}]"})
        for entity in PII_ENTITIES
    }
    result = anonymizer.anonymize(text=text, analyzer_results=findings, operators=operators)
    metrics.record_timing("redact_ms", (time.perf_counter() - start) * 1000)
    for finding in findings:
        metrics.increment(f"pii_redacted_{finding.entity_type.lower()}")
    metrics.increment("pii_redacted_total", len(findings))
    return result.text
