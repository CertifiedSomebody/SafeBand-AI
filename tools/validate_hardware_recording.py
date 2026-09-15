from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ai.hardware_schema import REQUIRED_COLUMNS, validate_columns, validate_row


def main():
    ap=argparse.ArgumentParser(description="Validate a SafeBand canonical hardware CSV.")
    ap.add_argument("--input",required=True)
    ap.add_argument("--report",default="hardware_validation_report.json")
    args=ap.parse_args()
    errors=[]; rows=0; sensor_counts={}; recording_counts={}
    with open(args.input,newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        missing=validate_columns(reader.fieldnames or [])
        if missing: errors.append({"row":0,"errors":[f"missing_column:{x}" for x in missing]})
        for i,row in enumerate(reader,2):
            rows+=1
            e=validate_row(row)
            if e and len(errors)<100: errors.append({"row":i,"errors":e})
            sensor_counts[row.get("sensor_type","")]=sensor_counts.get(row.get("sensor_type",""),0)+1
            recording_counts[row.get("recording_type","")]=recording_counts.get(row.get("recording_type",""),0)+1
    report={"valid":not errors,"rows":rows,"sensor_counts":sensor_counts,"recording_counts":recording_counts,"errors":errors[:100]}
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if not errors else 1)

if __name__=="__main__": main()
