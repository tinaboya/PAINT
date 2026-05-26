"""
PAINT BRIDGE inference.

Runs the *same* pain + demographic extraction as the WorldMedQA pipeline
(identical prompts, models, and vLLM serving), but over BRIDGE clinical-text
records instead of exam-question CSVs.

Input
-----
A BRIDGE JSON shaped as  {task_name: [record, ...]}  where each record has at
least `id`, `language`, and `input` (the clinical text). The repository ships
no data; obtain BRIDGE separately, e.g. data/raw/BRIDGE.example.all.json.

Output
------
    BRIDGE_result/{pain|demo}.{lang}.{task}/{bridge_model}/results.json

Each results.json is a list of {id, input, language, pred} records, where
`pred` is the raw model output. This is exactly the layout parse_bridge.py
consumes, so the follow-up is:

    python run_inference_bridge.py --input data/raw/BRIDGE.example.all.json
    python parse_bridge.py            # → model_outputs/<model>/BRIDGE_*.csv

Only the three ensemble models that were run on BRIDGE are configured here
(DeepSeek-R1-Qwen-32B, Qwen3-8B, Gemma4-E4B); the BRIDGE-side model-folder
names match parse_bridge.MODEL_NAME_MAP.

Usage:
    python run_inference_bridge.py                       # all tasks, 3 models
    python run_inference_bridge.py --model gemma4-e4b
    python run_inference_bridge.py --task Cantemis-NER
    python run_inference_bridge.py --dry-run
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm as async_tqdm

from config import CONCURRENCY, MODELS, PROMPT_FILES, RESULTS_DIR  # noqa: F401
from parse_utils import load_prompts
# Reuse the WorldMedQA server lifecycle + request code verbatim.
from run_inference import (
    call_vllm,
    start_vllm_server,
    stop_vllm_server,
    wait_for_vllm,
)
# Reuse the same BRIDGE model-folder naming the parser expects.
from parse_bridge import MODEL_NAME_MAP

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_OUT = REPO_ROOT / "BRIDGE_result"
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "BRIDGE.example.all.json"

# BRIDGE `language` value → prompt lang_code (SYSTEM_PROMPT_<CODE>).
# Brateca is Brazilian Portuguese, which uses the BZ prompt.
LANG_MAP = {"en": "EN", "es": "ES", "ja": "JP", "pt": "BZ"}

# config key → BRIDGE_result model-folder name (inverse of MODEL_NAME_MAP).
# Only the models that were run on BRIDGE appear here.
MODEL_KEY_TO_BRIDGE_DIR = {v: k for k, v in MODEL_NAME_MAP.items()}

# folder prefix per task kind (matches parse_bridge's "{pain|demo}." split)
KIND_PREFIX = {"pain": "pain", "demographics": "demo"}


def _output_path(kind: str, lang: str, task: str, bridge_model_dir: str) -> Path:
    return (BRIDGE_OUT / f"{KIND_PREFIX[kind]}.{lang}.{task}"
            / bridge_model_dir / "results.json")


async def _run_task(
    session: aiohttp.ClientSession,
    records: list,
    kind: str,
    system_prompt_template: str,
    user_message: str,
    model_hf_id: str,
    semaphore: asyncio.Semaphore,
    desc: str,
) -> list:
    """Run one extraction kind over all records of one task; return result list."""
    async def process(rec: dict) -> dict:
        text = str(rec.get("input", ""))
        system_prompt = system_prompt_template.format(text=text)
        raw = await call_vllm(session, system_prompt, user_message, model_hf_id, semaphore)
        return {"id": rec.get("id"), "input": text,
                "language": rec.get("language"), "pred": raw}

    coros = [process(rec) for rec in records]
    out = []
    for coro in async_tqdm.as_completed(coros, total=len(coros), desc=desc, unit="rec"):
        out.append(await coro)
    return out


async def run_model_on_bridge(
    model_key: str,
    model_cfg: dict,
    bridge_data: dict,
    prompts: dict,
    task_filter: str | None,
    force: bool,
    dry_run: bool,
) -> None:
    bridge_model_dir = MODEL_KEY_TO_BRIDGE_DIR[model_key]
    user_message = model_cfg.get("user_message", "Please analyze the text above.")

    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(CONCURRENCY)

        for task, records in bridge_data.items():
            if task_filter and task != task_filter:
                continue
            if not records:
                continue
            # BRIDGE records within a task share one language in practice;
            # group defensively in case they don't.
            by_lang: dict[str, list] = {}
            for rec in records:
                by_lang.setdefault(rec.get("language", ""), []).append(rec)

            for raw_lang, recs in by_lang.items():
                lang = LANG_MAP.get(raw_lang)
                if lang is None:
                    logger.warning("[%s|%s] no prompt for language=%r — skipping %d recs.",
                                   model_key, task, raw_lang, len(recs))
                    continue

                for kind in ("pain", "demographics"):
                    out_path = _output_path(kind, lang, task, bridge_model_dir)
                    if out_path.exists() and not force:
                        logger.info("[%s|%s|%s] exists, skip (use --force): %s",
                                    model_key, task, kind, out_path.relative_to(REPO_ROOT))
                        continue
                    if dry_run:
                        logger.info("[DRY-RUN] %s  %d recs → %s",
                                    kind, len(recs), out_path.relative_to(REPO_ROOT))
                        continue

                    results = await _run_task(
                        session=session,
                        records=recs,
                        kind=kind,
                        system_prompt_template=prompts[kind][lang],
                        user_message=user_message,
                        model_hf_id=model_cfg["hf_id"],
                        semaphore=semaphore,
                        desc=f"{model_key}|{task}|{lang}|{kind}",
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    logger.info("[%s|%s|%s] wrote %d → %s",
                                model_key, task, kind, len(results),
                                out_path.relative_to(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="PAINT BRIDGE inference (3 ensemble models)")
    ap.add_argument("--input", default=str(DEFAULT_INPUT),
                    help="BRIDGE JSON {task: [records]} (default: %(default)s)")
    ap.add_argument("--model", default=None,
                    help=f"Model key to run (default: all). Choices: {list(MODEL_KEY_TO_BRIDGE_DIR)}")
    ap.add_argument("--task", default=None, help="Run only this BRIDGE task name")
    ap.add_argument("--force", action="store_true", help="Overwrite existing results.json")
    ap.add_argument("--dry-run", action="store_true", help="Report what would run; no API calls")
    ap.add_argument("--skip-server", action="store_true",
                    help="Assume vLLM is already running (don't start/stop it)")
    args = ap.parse_args()

    if args.model and args.model not in MODEL_KEY_TO_BRIDGE_DIR:
        logger.error("Unknown/unsupported model '%s'. BRIDGE models: %s",
                     args.model, list(MODEL_KEY_TO_BRIDGE_DIR))
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        bridge_data = json.load(f)
    logger.info("Loaded BRIDGE input: %d tasks from %s", len(bridge_data), args.input)

    prompts = {kind: load_prompts(pf) for kind, pf in PROMPT_FILES.items()}

    models_to_run = ([args.model] if args.model
                     else [k for k in MODELS if k in MODEL_KEY_TO_BRIDGE_DIR])

    for model_key in models_to_run:
        model_cfg = MODELS[model_key]
        logger.info("=" * 60)
        logger.info("BRIDGE | Model: %s  (%s)", model_key, model_cfg["hf_id"])
        logger.info("=" * 60)

        proc = None
        try:
            if not args.dry_run and not args.skip_server:
                proc = start_vllm_server(model_cfg, model_key)
                wait_for_vllm(proc, model_key)
            asyncio.run(run_model_on_bridge(
                model_key, model_cfg, bridge_data, prompts,
                args.task, args.force, args.dry_run))
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            sys.exit(0)
        except Exception as exc:
            logger.exception("Error running model %s on BRIDGE: %s", model_key, exc)
        finally:
            if proc is not None and not args.skip_server:
                stop_vllm_server(proc, model_key)

    logger.info("All done. Next: python parse_bridge.py")


if __name__ == "__main__":
    main()
