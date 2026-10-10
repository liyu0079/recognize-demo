"""Run the local visual-regression manifest against a running API service.

Manifest schema::

  {"cases": [{"id": "dense-tomatoes", "image": "dense-tomatoes/a.jpg",
               "objects": [{"label": "tomato", "bbox": [x1,y1,x2,y2]}],
               "ocr_texts": ["..."], "mask_required": ["tomato"]}]}

All paths are relative to the manifest.  The script intentionally uses only
the standard library so it can run in the offline deployment environment.
"""
# // 优化：本地回归体系 - 固定标注集可重复计算召回、误检、OCR、mask、时延与资源指标。
# // 优化：本地回归体系 - 固定标注集可重复计算召回、误检、OCR、mask、时延与资源指标。
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import statistics
import time
import uuid
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1]) + max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1]) - intersection
    return intersection / union if union else 0.0


def multipart(image: Path, run_mode: str) -> tuple[bytes, str]:
    boundary = f"----vision-regression-{uuid.uuid4().hex}"
    kind = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
    body = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"text_prompt\"\r\n\r\n\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"prompt_mode\"\r\n\r\nuniversal\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"run_mode\"\r\n\r\n{run_mode}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"media\"; filename=\"{image.name}\"\r\nContent-Type: {kind}\r\n\r\n".encode(),
        image.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    return b"".join(body), boundary


def analyze(url: str, image: Path, run_mode: str) -> dict:
    body, boundary = multipart(image, run_mode)
    request = Request(url.rstrip("/") + "/api/analyze-media", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def score_case(case: dict, response: dict) -> dict[str, object]:
    expected = case.get("objects", [])
    predicted = [item for item in response.get("objects", []) if item.get("label") not in {"text", "sign text", "license plate"}]
    matched: set[int] = set()
    true_positive = 0
    for target in expected:
        label = str(target.get("label", "")).lower()
        bbox = target.get("bbox", [])
        match = next((index for index, item in enumerate(predicted) if index not in matched and item.get("label", "").lower() == label and len(bbox) == 4 and iou(bbox, item.get("bbox", [])) >= 0.5), None)
        if match is not None:
            matched.add(match)
            true_positive += 1
    expected_text = {str(text).strip() for text in case.get("ocr_texts", []) if str(text).strip()}
    actual_text = {str(item.get("ocr_text", "")).strip() for item in response.get("objects", []) if str(item.get("ocr_text", "")).strip()}
    mask_labels = {str(label).lower() for label in case.get("mask_required", [])}
    mask_total = sum(1 for item in response.get("objects", []) if item.get("label", "").lower() in mask_labels)
    mask_ready = sum(1 for item in response.get("objects", []) if item.get("label", "").lower() in mask_labels and item.get("mask_rle"))
    return {
        "id": case.get("id", "unnamed"),
        "gt": len(expected), "tp": true_positive, "fp": max(0, len(predicted) - len(matched)),
        "recall": round(true_positive / len(expected), 4) if expected else "",
        "false_positive_rate": round(max(0, len(predicted) - len(matched)) / len(predicted), 4) if predicted else 0.0,
        "object_count": len(predicted),
        "ocr_accuracy": round(len(expected_text & actual_text) / len(expected_text), 4) if expected_text else "",
        "mask_coverage": round(mask_ready / mask_total, 4) if mask_total else "",
        "total_ms": response.get("timings_ms", {}).get("total", ""),
        "dino_ms": response.get("timings_ms", {}).get("dino_detection", ""),
        "sam2_ms": response.get("timings_ms", {}).get("sam2_segmentation", ""),
        "ocr_ms": response.get("timings_ms", {}).get("ocr", ""),
        "memory_peak_mb": response.get("resource_usage", {}).get("memory_peak_mb", ""),
        "process_peak_rss_mb": response.get("resource_usage", {}).get("process_peak_rss_mb", ""),
        "cpu_process_ms": response.get("resource_usage", {}).get("cpu_process_ms", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local vision API against a labelled regression manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-mode", choices=("fast", "full"), default="full")
    parser.add_argument("--output", type=Path, default=Path("regression-results.csv"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    results: list[dict[str, object]] = []
    for case in cases:
        image = (args.manifest.parent / case["image"]).resolve()
        started = time.perf_counter()
        try:
            result = score_case(case, analyze(args.url, image, args.run_mode))
            result["wall_ms"] = round((time.perf_counter() - started) * 1000, 2)
            results.append(result)
            print(f"[ok] {result['id']}: recall={result['recall']} total_ms={result['total_ms']}")
        except (OSError, HTTPError, URLError, ValueError) as exc:
            results.append({"id": case.get("id", "unnamed"), "error": str(exc)})
            print(f"[failed] {case.get('id', 'unnamed')}: {exc}")
    fields = sorted({key for result in results for key in result})
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    total_values = [float(row["total_ms"]) for row in results if isinstance(row.get("total_ms"), (int, float))]
    if total_values:
        ordered = sorted(total_values)
        print(f"[summary] average_ms={statistics.mean(ordered):.2f} p95_ms={ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.95))]:.2f}")


if __name__ == "__main__":
    main()
