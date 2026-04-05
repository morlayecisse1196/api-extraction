from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from PIL import Image

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from models_loader.loader import ModelRegistry
from services.mapper import map_predictions
from services.pipeline import run_pipeline_with_profile


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Benchmark baseline extraction (10 images) avec timings par etape.",
	)
	parser.add_argument("--images-dir", required=True, help="Dossier contenant les images de test.")
	parser.add_argument("--limit", type=int, default=10, help="Nombre d'images a evaluer.")
	parser.add_argument("--report-dir", default="reports", help="Dossier de sortie des rapports.")
	parser.add_argument("--recursive", action="store_true", help="Recherche recursive des images.")
	return parser.parse_args()


def list_images(images_dir: Path, recursive: bool) -> list[Path]:
	iterator = images_dir.rglob("*") if recursive else images_dir.glob("*")
	files = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
	return sorted(files)


def pct95(values: list[float]) -> float:
	if not values:
		return 0.0
	return float(np.percentile(np.array(values, dtype=np.float64), 95))


def summarize(values: list[float]) -> dict:
	if not values:
		return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
	return {
		"count": len(values),
		"mean": round(float(mean(values)), 4),
		"min": round(float(min(values)), 4),
		"max": round(float(max(values)), 4),
		"p95": round(float(pct95(values)), 4),
	}


def get_env_info() -> dict:
	cuda_available = torch.cuda.is_available()
	device_count = torch.cuda.device_count()
	return {
		"python": platform.python_version(),
		"platform": platform.platform(),
		"torch": torch.__version__,
		"cuda_available": cuda_available,
		"torch_cuda_version": torch.version.cuda,
		"cuda_device_count": device_count,
		"cuda_device_name": torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else None,
	}


def to_markdown(report: dict) -> str:
	env = report["environment"]
	s = report["stats"]
	lines = []
	lines.append("# Rapport baseline API extraction")
	lines.append("")
	lines.append(f"- Date: {report['generated_at']}")
	lines.append(f"- Images analysees: {report['images_evaluated']} / {report['images_requested']}")
	lines.append(f"- Dossier source: {report['images_dir']}")
	lines.append("")
	lines.append("## Environnement inference")
	lines.append("")
	lines.append(f"- Python: {env['python']}")
	lines.append(f"- Torch: {env['torch']}")
	lines.append(f"- CUDA dispo: {env['cuda_available']}")
	lines.append(f"- CUDA torch: {env['torch_cuda_version']}")
	lines.append(f"- GPU count: {env['cuda_device_count']}")
	lines.append(f"- GPU name: {env['cuda_device_name']}")
	lines.append("")
	lines.append("## Statistiques (secondes)")
	lines.append("")
	lines.append("| Etape | count | mean | min | max | p95 |")
	lines.append("|---|---:|---:|---:|---:|---:|")
	for key in ["total", "rectification", "yolo", "trocr", "layoutlm", "mapping"]:
		row = s[key]
		lines.append(
			f"| {key} | {row['count']} | {row['mean']} | {row['min']} | {row['max']} | {row['p95']} |"
		)
	lines.append("")
	lines.append("## Notes")
	lines.append("")
	if env["cuda_available"]:
		lines.append("- Inference GPU activee. Verifier charge GPU avec nvidia-smi pendant benchmark.")
	else:
		lines.append("- Inference CPU uniquement. Latence elevee attendue sur TrOCR/LayoutLMv3.")
	if report["errors"]:
		lines.append(f"- Erreurs image: {len(report['errors'])}")
	return "\n".join(lines)


def main() -> None:
	args = parse_args()
	images_dir = Path(args.images_dir).resolve()
	if not images_dir.exists() or not images_dir.is_dir():
		raise SystemExit(f"Dossier introuvable: {images_dir}")

	candidates = list_images(images_dir, args.recursive)
	selected = candidates[: max(0, args.limit)]
	if not selected:
		raise SystemExit("Aucune image trouvee dans le dossier fourni.")

	report_dir = (ROOT / args.report_dir).resolve()
	report_dir.mkdir(parents=True, exist_ok=True)

	t_load = perf_counter()
	registry = ModelRegistry()
	registry.load_all()
	load_time_s = round(perf_counter() - t_load, 4)

	per_image = []
	errors = []

	totals = []
	rect_s = []
	yolo_s = []
	trocr_s = []
	layout_s = []
	mapping_s = []

	for img_path in selected:
		try:
			pil_img = Image.open(img_path).convert("RGB")

			t_total = perf_counter()
			predictions, profile = run_pipeline_with_profile(pil_img, registry)

			t_map = perf_counter()
			structured = map_predictions(predictions)
			map_t = perf_counter() - t_map

			total_t = perf_counter() - t_total

			timings = profile["timings_s"]
			totals.append(total_t)
			rect_s.append(float(timings["rectification"]))
			yolo_s.append(float(timings["yolo"]))
			trocr_s.append(float(timings["trocr"]))
			layout_s.append(float(timings["layoutlm"]))
			mapping_s.append(map_t)

			per_image.append(
				{
					"image": str(img_path),
					"total_s": round(total_t, 4),
					"rectification_s": round(float(timings["rectification"]), 4),
					"yolo_s": round(float(timings["yolo"]), 4),
					"trocr_s": round(float(timings["trocr"]), 4),
					"layoutlm_s": round(float(timings["layoutlm"]), 4),
					"mapping_s": round(map_t, 4),
					"detections": int(profile["counts"]["detections"]),
					"tokens": int(profile["counts"]["tokens"]),
					"trocr_calls": int(profile["counts"]["trocr_calls"]),
					"rectification": profile["rectification"],
					"fields_extracted": len([k for k in structured.keys() if k != "_meta"]),
				}
			)
		except Exception as exc:
			errors.append({"image": str(img_path), "error": f"{type(exc).__name__}: {exc}"})

	report = {
		"generated_at": datetime.now().isoformat(timespec="seconds"),
		"images_dir": str(images_dir),
		"images_requested": int(args.limit),
		"images_evaluated": len(per_image),
		"model_load_time_s": load_time_s,
		"environment": get_env_info(),
		"stats": {
			"total": summarize(totals),
			"rectification": summarize(rect_s),
			"yolo": summarize(yolo_s),
			"trocr": summarize(trocr_s),
			"layoutlm": summarize(layout_s),
			"mapping": summarize(mapping_s),
		},
		"per_image": per_image,
		"errors": errors,
	}

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	json_path = report_dir / f"baseline_report_{timestamp}.json"
	md_path = report_dir / f"baseline_report_{timestamp}.md"

	json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
	md_path.write_text(to_markdown(report), encoding="utf-8")

	print(f"Rapport JSON: {json_path}")
	print(f"Rapport MD:   {md_path}")
	print(f"Images evaluees: {len(per_image)}")
	print(f"Mode inference: {'GPU' if report['environment']['cuda_available'] else 'CPU'}")


if __name__ == "__main__":
	main()
