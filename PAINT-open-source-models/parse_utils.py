"""
Utilities for parsing model outputs:
  - strip_thinking(): remove <think>...</think> blocks (DeepSeek-R1, Qwen3-Thinking)
  - load_prompts(): exec the prompt .txt files and return the SYSTEM_PROMPT_* variables
  - extract_final_json(): locate FINAL_JSON: delimiter and parse the JSON object
  - validate_pain_json(): enforce hierarchical constraints
  - validate_demo_json(): check required keys are present
"""

import json
import re
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thinking-block stripper
# ---------------------------------------------------------------------------

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models."""
    return _THINK_PATTERN.sub("", text).strip()


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def load_prompts(prompt_file: str) -> dict[str, str]:
    """
    Execute a prompt .txt file that contains Python variable assignments like:
        SYSTEM_PROMPT_EN = \"\"\"...\"\"\"\
        SYSTEM_PROMPT_JP = \"\"\"...\"\"\"\
        ...
    Returns a dict: {"EN": "...", "JP": "...", "ES": "...", "BZ": "..."}
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        source = f.read()

    namespace: dict = {}
    exec(compile(source, prompt_file, "exec"), namespace)  # noqa: S102

    prompts = {}
    for key, value in namespace.items():
        if key.startswith("SYSTEM_PROMPT_") and isinstance(value, str):
            lang_code = key[len("SYSTEM_PROMPT_"):]
            prompts[lang_code] = value

    if not prompts:
        raise ValueError(f"No SYSTEM_PROMPT_* variables found in {prompt_file}")

    return prompts


# ---------------------------------------------------------------------------
# FINAL_JSON extractor
# ---------------------------------------------------------------------------

_FINAL_JSON_MARKER = "FINAL_JSON:"


def extract_final_json(text: str) -> dict | None:
    """
    Locate the FINAL_JSON: marker in the model output and parse the JSON that follows.
    Returns the parsed dict, or None if the marker or valid JSON is not found.
    """
    # Strip thinking blocks first
    cleaned = strip_thinking(text)

    marker_pos = cleaned.find(_FINAL_JSON_MARKER)
    if marker_pos == -1:
        logger.debug("FINAL_JSON: marker not found in output")
        return None

    after_marker = cleaned[marker_pos + len(_FINAL_JSON_MARKER):].strip()

    # Find the outermost JSON object
    brace_start = after_marker.find("{")
    if brace_start == -1:
        logger.debug("No opening brace found after FINAL_JSON: marker")
        return None

    # Walk to find the matching closing brace
    depth = 0
    brace_end = -1
    for i, ch in enumerate(after_marker[brace_start:], start=brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                brace_end = i + 1
                break

    if brace_end == -1:
        logger.debug("Could not find matching closing brace for JSON object")
        return None

    json_str = after_marker[brace_start:brace_end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        # Fallback: some models (e.g. Qwen3-Thinking on Brateca) emit
        # LaTeX-style double braces "{{...}}" inside FINAL_JSON. Try
        # collapsing them and re-parsing.
        if "{{" in json_str or "}}" in json_str:
            collapsed = json_str.replace("{{", "{").replace("}}", "}")
            try:
                return json.loads(collapsed)
            except json.JSONDecodeError:
                pass
        logger.debug("JSON decode error: %s | snippet: %.200s", exc, json_str)
        return None


# ---------------------------------------------------------------------------
# Schema validators
# ---------------------------------------------------------------------------

_DEMO_REQUIRED_KEYS = {"age", "gender", "ethnicity", "occupation"}

_PAIN_REQUIRED_KEYS = {
    "pain_related", "pain_presence", "location",
    "pain_type", "severity", "is_mental", "context",
}


def validate_demo_json(obj: dict) -> list[str]:
    """
    Validate a demographics extraction result.
    Returns a list of error strings (empty list = valid).
    """
    errors = []
    missing = _DEMO_REQUIRED_KEYS - obj.keys()
    if missing:
        errors.append(f"Missing keys: {missing}")
    return errors


def validate_pain_json(obj: dict) -> list[str]:
    """
    Validate a pain extraction result including hierarchical constraints.
    Returns a list of error strings (empty list = valid).
    """
    errors = []

    missing = _PAIN_REQUIRED_KEYS - obj.keys()
    if missing:
        errors.append(f"Missing keys: {missing}")
        return errors  # can't check constraints without the keys

    pain_related = obj.get("pain_related")
    pain_presence = obj.get("pain_presence")
    location = obj.get("location")
    pain_type = obj.get("pain_type")

    # Constraint 1: pain_related=false → pain_presence must be false
    if pain_related is False and pain_presence is not False:
        errors.append("pain_related=false but pain_presence is not false")

    # Constraint 2: pain_presence=false → location must be empty list
    if pain_presence is False and location not in ([], None):
        errors.append("pain_presence=false but location is not empty")

    # Constraint 3: pain_presence=false → pain_type must be empty list
    if pain_presence is False and pain_type not in ([], None):
        errors.append("pain_presence=false but pain_type is not empty")

    return errors


# ---------------------------------------------------------------------------
# Convenience: parse + validate in one call
# ---------------------------------------------------------------------------

def parse_demographics(raw_output: str) -> tuple[dict | None, str]:
    """
    Parse and validate a demographics extraction response.
    Returns (parsed_dict_or_None, error_string).
    error_string is empty on success.
    """
    obj = extract_final_json(raw_output)
    if obj is None:
        return None, "FINAL_JSON not found or not parseable"

    errors = validate_demo_json(obj)
    if errors:
        return obj, "; ".join(errors)

    return obj, ""


def parse_pain(raw_output: str) -> tuple[dict | None, str]:
    """
    Parse and validate a pain extraction response.
    Returns (parsed_dict_or_None, error_string).
    error_string is empty on success.
    """
    obj = extract_final_json(raw_output)
    if obj is None:
        return None, "FINAL_JSON not found or not parseable"

    errors = validate_pain_json(obj)
    if errors:
        return obj, "; ".join(errors)

    return obj, ""
