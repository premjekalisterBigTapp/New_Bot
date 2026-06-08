from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv, find_dotenv

# Ensure .env is loaded before reading any config
load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

# --- Paths ---
# Agentic layer uses its own configs folder to avoid modifying legacy configs
# config.py is in bigtapp/src/bigtapp/agentic/
# Path(__file__).parent is bigtapp/src/bigtapp/agentic/
AGENTIC_DIR = Path(__file__).resolve().parent
CONFIG_DIR = AGENTIC_DIR / "configs"

IR_RESPONSE_PATH = CONFIG_DIR / "ir_response.yaml"
SUMMARY_RESPONSE_PATH = CONFIG_DIR / "summary_response.yaml"
CMP_RESPONSE_PATH = CONFIG_DIR / "cmp_response.yaml"
RECOMMENDATION_RESPONSE_PATH = CONFIG_DIR / "recommendation_response.yaml"
SLOT_RULES_PATH = CONFIG_DIR / "slot_validation_rules.yaml"
KNOWLEDGE_BASE_PATH = CONFIG_DIR / "knowledge_base.txt"
LINKS_PATH = CONFIG_DIR / "links.yaml"
LIFE_INSURANCE_KB_PATH = CONFIG_DIR / "life_insurance_kb.txt"

# --- Caches ---
_ir_templates_cache: Dict[str, Any] = {}
_summary_templates_cache: Dict[str, Any] = {}
_cmp_templates_cache: Dict[str, Any] = {}
_rec_templates_cache: Dict[str, Any] = {}
_slot_rules_cache: Dict[str, Any] = {}
_links_cache: Dict[str, str] = {}
_kb_text_cache: Optional[str] = None
_life_kb_cache: Optional[str] = None


def _load_yaml_cached(path: Path, cache: Dict[str, Any]) -> Dict[str, Any]:
    if cache:
        return cache
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if isinstance(data, dict):
            cache.update({str(k).lower(): v for k, v in data.items()})
        else:
            cache.clear()
    except Exception as e:
        logger.warning("Agentic: failed to load YAML from %s - %s", path, e)
        cache.clear()
    return cache


def _load_ir_templates() -> Dict[str, Any]:
    return _load_yaml_cached(IR_RESPONSE_PATH, _ir_templates_cache)


def _load_summary_templates() -> Dict[str, Any]:
    return _load_yaml_cached(SUMMARY_RESPONSE_PATH, _summary_templates_cache)


def _load_cmp_templates() -> Dict[str, Any]:
    return _load_yaml_cached(CMP_RESPONSE_PATH, _cmp_templates_cache)


def _load_rec_templates() -> Dict[str, Any]:
    return _load_yaml_cached(RECOMMENDATION_RESPONSE_PATH, _rec_templates_cache)


def _load_slot_rules() -> Dict[str, Any]:
    if _slot_rules_cache:
        return _slot_rules_cache
    try:
        text = SLOT_RULES_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if isinstance(data, dict):
            _slot_rules_cache.update(data)
    except Exception as e:
        logger.warning("Agentic: failed to load slot_validation_rules.yaml - %s", e)
    return _slot_rules_cache


def _load_knowledge_base() -> str:
    global _kb_text_cache
    if isinstance(_kb_text_cache, str):
        return _kb_text_cache
    try:
        _kb_text_cache = KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")
    except Exception:
        _kb_text_cache = ""
    return _kb_text_cache


def _load_purchase_links() -> Dict[str, str]:
    if _links_cache:
        return _links_cache
    try:
        text = LINKS_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if isinstance(data, dict):
            _links_cache.update({str(k).lower(): str(v) for k, v in data.items()})
    except Exception as e:
        logger.warning("Agentic: failed to load links.yaml - %s", e)
    return _links_cache


def _load_life_insurance_kb() -> str:
    """Load the brand-neutral life-insurance knowledge base text (cached).

    This is the source of truth for the `life_insurance` intent. Returns an
    empty string if the file is missing.
    """
    global _life_kb_cache
    if isinstance(_life_kb_cache, str):
        return _life_kb_cache
    try:
        _life_kb_cache = LIFE_INSURANCE_KB_PATH.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Agentic: failed to load life_insurance_kb.txt - %s", e)
        _life_kb_cache = ""
    return _life_kb_cache
