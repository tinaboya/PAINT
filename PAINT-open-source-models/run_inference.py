"""
PAINT batch inference script.

Runs the five open-weight models defined in config.MODELS over every
dataset, writing one CSV per (model, dataset, task) into results/.

Usage:
    # Run all five models sequentially:
    python run_inference.py

    # Run a specific model only:
    python run_inference.py --model deepseek-r1-qwen-32b

    # Run a specific model on a specific dataset:
    python run_inference.py --model gemma4-e4b --dataset Brazil

    # Dry-run (print what would be processed, skip API calls):
    python run_inference.py --dry-run

The script:
  1. Starts a vLLM server for the current model
  2. For each dataset, loads the CSV and runs both extraction tasks
     (demographics + pain) using the language-matched system prompt
  3. Saves results to results/{model_name}/{Dataset}_{task}.csv
  4. Supports checkpoint/resume: rows already in the output CSV are skipped
  5. Shuts down the vLLM server before loading the next model
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm

from config import (
    CONCURRENCY,
    DATASETS,
    GENERATION_CONFIG,
    MODELS,
    PROMPT_FILES,
    RESULTS_DIR,
    VLLM_HEALTH_CHECK_INTERVAL,
    VLLM_HOST,
    VLLM_PORT,
    VLLM_STARTUP_TIMEOUT,
)
from parse_utils import load_prompts, parse_demographics, parse_pain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VLLM_API_URL = f"{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions"
VLLM_HEALTH_URL = f"{VLLM_HOST}:{VLLM_PORT}/health"

TASK_PARSERS = {
    "demographics": parse_demographics,
    "pain": parse_pain,
}

DEMO_FIELDS = ["age", "gender", "ethnicity", "occupation"]
PAIN_FIELDS = ["pain_related", "pain_presence", "location", "pain_type", "severity", "is_mental", "context"]

TASK_FIELDS = {
    "demographics": DEMO_FIELDS,
    "pain": PAIN_FIELDS,
}


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def start_vllm_server(model_cfg: dict, model_name: str) -> subprocess.Popen:
    """Launch vLLM server as a subprocess and return the Popen handle."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_cfg["hf_id"],
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--disable-log-requests",
    ] + model_cfg.get("vllm_extra_args", [])

    logger.info("[%s] Starting vLLM server: %s", model_name, " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_vllm(proc: subprocess.Popen, model_name: str) -> None:
    """Poll the /health endpoint until vLLM is ready or timeout."""
    import urllib.request
    deadline = time.time() + VLLM_STARTUP_TIMEOUT
    while time.time() < deadline:
        # Check if the process crashed
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM process exited unexpectedly (code {proc.returncode})")
        try:
            with urllib.request.urlopen(VLLM_HEALTH_URL, timeout=2):
                logger.info("[%s] vLLM server is ready.", model_name)
                return
        except Exception:
            pass
        time.sleep(VLLM_HEALTH_CHECK_INTERVAL)
    raise TimeoutError(f"vLLM did not become healthy within {VLLM_STARTUP_TIMEOUT}s")


def stop_vllm_server(proc: subprocess.Popen, model_name: str) -> None:
    """Gracefully terminate the vLLM server subprocess."""
    if proc.poll() is None:
        logger.info("[%s] Stopping vLLM server (pid=%d)…", model_name, proc.pid)
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    logger.info("[%s] vLLM server stopped.", model_name)


# ---------------------------------------------------------------------------
# Async inference
# ---------------------------------------------------------------------------

async def call_vllm(
    session: aiohttp.ClientSession,
    system_prompt: str,
    user_message: str,
    model_hf_id: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> str:
    """
    Call the vLLM OpenAI-compatible chat completions endpoint.
    Returns the raw response text (assistant message content).
    """
    payload = {
        "model": model_hf_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        **GENERATION_CONFIG,
    }

    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                async with session.post(VLLM_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    body = await resp.text()
                    logger.warning("HTTP %d on attempt %d: %.200s", resp.status, attempt, body)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("Request error on attempt %d: %s", attempt, exc)

        if attempt < max_retries:
            await asyncio.sleep(2 ** attempt)

    return ""  # empty string signals a failed call


async def run_task_for_dataset(
    df: pd.DataFrame,
    task: str,
    system_prompt_template: str,
    user_message: str,
    model_hf_id: str,
    output_path: Path,
    model_name: str,
    dataset_name: str,
) -> None:
    """
    Run one extraction task (demographics or pain) over all rows in df.
    Results are appended to output_path row-by-row (checkpoint-safe).
    """
    parser = TASK_PARSERS[task]
    fields = TASK_FIELDS[task]

    # Load already-completed Question_IDs for checkpointing
    completed_ids: set = set()
    if output_path.exists():
        existing = pd.read_csv(output_path, usecols=["Question_ID"])
        completed_ids = set(existing["Question_ID"].astype(str))
        logger.info("[%s|%s|%s] Resuming — %d rows already done.",
                    model_name, dataset_name, task, len(completed_ids))

    rows_to_process = df[~df["Question_ID"].astype(str).isin(completed_ids)]
    if rows_to_process.empty:
        logger.info("[%s|%s|%s] All rows already processed, skipping.", model_name, dataset_name, task)
        return

    # Open output CSV in append mode
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    out_file = open(output_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        out_file,
        fieldnames=["Question_ID", "model", "dataset", "raw_response", "parse_error"] + fields,
        extrasaction="ignore",
    )
    if write_header:
        csv_writer.writeheader()

    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY + 4)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def process_row(row: pd.Series) -> dict:
            q_id = str(row["Question_ID"])
            question_text = str(row.get("Question", ""))
            system_prompt = system_prompt_template.format(text=question_text)

            raw = await call_vllm(session, system_prompt, user_message, model_hf_id, semaphore)

            parsed, error = parser(raw) if raw else (None, "empty response from model")

            result = {
                "Question_ID":  q_id,
                "model":        model_name,
                "dataset":      dataset_name,
                "raw_response": raw,
                "parse_error":  error,
            }
            if parsed:
                for field in fields:
                    val = parsed.get(field)
                    # Serialise lists to JSON strings for CSV storage
                    result[field] = json.dumps(val, ensure_ascii=False) if isinstance(val, list) else val
            else:
                for field in fields:
                    result[field] = None

            return result

        tasks = [process_row(row) for _, row in rows_to_process.iterrows()]

        # Process with progress bar; write results as they complete
        for coro in async_tqdm.as_completed(
            tasks,
            total=len(tasks),
            desc=f"{model_name}|{dataset_name}|{task}",
            unit="row",
        ):
            result = await coro
            csv_writer.writerow(result)
            out_file.flush()

    out_file.close()
    logger.info("[%s|%s|%s] Done. Output: %s", model_name, dataset_name, task, output_path)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def run_model(model_name: str, model_cfg: dict, datasets: dict, prompts: dict, dry_run: bool) -> None:
    """Run all datasets and tasks for a single model."""
    results_model_dir = Path(RESULTS_DIR) / model_name
    results_model_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_cfg in datasets.items():
        lang_code = dataset_cfg["lang_code"]
        csv_path = dataset_cfg["csv_path"]

        logger.info("[%s|%s] Loading CSV: %s", model_name, dataset_name, csv_path)
        df = pd.read_csv(csv_path)
        logger.info("[%s|%s] %d rows loaded.", model_name, dataset_name, len(df))

        for task, prompt_dict in prompts.items():
            if lang_code not in prompt_dict:
                logger.warning("[%s|%s|%s] No prompt for lang_code=%s — skipping.",
                               model_name, dataset_name, task, lang_code)
                continue

            system_prompt_template = prompt_dict[lang_code]
            user_message = model_cfg.get("user_message", "Please analyze the text above.")
            output_path = results_model_dir / f"{dataset_name}_{task}.csv"

            if dry_run:
                logger.info("[DRY-RUN] Would process %d rows → %s", len(df), output_path)
                continue

            await run_task_for_dataset(
                df=df,
                task=task,
                system_prompt_template=system_prompt_template,
                user_message=user_message,
                model_hf_id=model_cfg["hf_id"],
                output_path=output_path,
                model_name=model_name,
                dataset_name=dataset_name,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="PAINT open-source model inference")
    parser.add_argument("--model", default=None,
                        help=f"Model key to run (default: all). Choices: {list(MODELS)}")
    parser.add_argument("--dataset", default=None,
                        help=f"Dataset to run (default: all). Choices: {list(DATASETS)}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without making any API calls")
    parser.add_argument("--skip-server", action="store_true",
                        help="Assume vLLM is already running (don't start/stop it)")
    args = parser.parse_args()

    # Filter models and datasets if specified
    models_to_run = {args.model: MODELS[args.model]} if args.model else MODELS
    datasets_to_run = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    if args.model and args.model not in MODELS:
        logger.error("Unknown model '%s'. Valid choices: %s", args.model, list(MODELS))
        sys.exit(1)
    if args.dataset and args.dataset not in DATASETS:
        logger.error("Unknown dataset '%s'. Valid choices: %s", args.dataset, list(DATASETS))
        sys.exit(1)

    # Load prompt templates once (shared across models)
    logger.info("Loading prompt templates…")
    prompts: dict[str, dict[str, str]] = {}
    for task, prompt_file in PROMPT_FILES.items():
        prompts[task] = load_prompts(prompt_file)
        logger.info("  %s: langs = %s", task, sorted(prompts[task].keys()))

    # Run models sequentially
    for model_name, model_cfg in models_to_run.items():
        logger.info("=" * 60)
        logger.info("Model: %s  (%s)", model_name, model_cfg["hf_id"])
        logger.info("=" * 60)

        proc = None
        try:
            if not args.dry_run and not args.skip_server:
                proc = start_vllm_server(model_cfg, model_name)
                wait_for_vllm(proc, model_name)

            asyncio.run(run_model(model_name, model_cfg, datasets_to_run, prompts, args.dry_run))

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            sys.exit(0)
        except Exception as exc:
            logger.exception("Error running model %s: %s", model_name, exc)
        finally:
            if proc is not None and not args.skip_server:
                stop_vllm_server(proc, model_name)

    logger.info("All done.")


if __name__ == "__main__":
    main()
