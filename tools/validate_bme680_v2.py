from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib


def main():
    ap = argparse.ArgumentParser(description="Validate BME680 AIRWISE V2 artifact/report contract.")
    ap.add_argument("--model", default="models/bme680_airwise_v2/bme680_airwise_v2.joblib")
    ap.add_argument("--report", default="models/bme680_airwise_v2/report.json")
    args = ap.parse_args()

    model_path = (ROOT / args.model).resolve()
    report_path = (ROOT / args.report).resolve()
    checks = {
        "model_exists": model_path.exists(),
        "report_exists": report_path.exists(),
    }
    if not checks["model_exists"] or not checks["report_exists"]:
        checks["status"] = "FAIL"
        print(json.dumps(checks, indent=2))
        raise SystemExit(1)

    artifact = joblib.load(model_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks.update({
        "artifact_version": artifact.get("version"),
        "feature_count": len(artifact.get("feature_columns", [])),
        "selected_model": report.get("selected_model"),
        "test_metrics_present": "test_metrics" in report,
        "test_used_for_selection": report.get("research_guardrails", {}).get("test_used_for_selection"),
        "iaq_proxy_used_as_input": report.get("research_guardrails", {}).get("iaQ_proxy_used_as_input", report.get("research_guardrails", {}).get("iaq_proxy_used_as_input")),
        "future_temporal_values_used": report.get("research_guardrails", {}).get("future_temporal_values_used"),
        "label_column_used_as_input": report.get("research_guardrails", {}).get("label_column_used_as_input"),
    })
    cols = set(artifact.get("feature_columns", []))
    forbidden = {"IAQ_proxy", "IAQ_proxy_zscore", "IAQ_class"}
    checks["forbidden_feature_columns_present"] = sorted(cols & forbidden)
    checks["status"] = "PASS" if (
        checks["test_metrics_present"]
        and checks["test_used_for_selection"] is False
        and checks["iaq_proxy_used_as_input"] is False
        and checks["future_temporal_values_used"] is False
        and checks["label_column_used_as_input"] is False
        and not checks["forbidden_feature_columns_present"]
    ) else "FAIL"
    print(json.dumps(checks, indent=2))
    if checks["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
