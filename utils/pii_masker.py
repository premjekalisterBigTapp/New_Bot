"""
Centralized PII Masker for BigTapp Agentic Chatbot
================================================

Enterprise-grade PII protection that masks sensitive data before ANY LLM processing.
All user messages pass through this masker at the entry point to ensure no PII
reaches the LLM.

Supported PII Types (Singapore context):
- NRIC/FIN: S1234567D format
- Mobile numbers: +65XXXXXXXX or 8/9XXXXXXX
- Email addresses
- Credit card numbers
- Policy numbers (BigTapp format: XX######)
- Postal codes (6 digits)
- Passport numbers

Usage:
    masker = get_pii_masker()
    masked_text, mapping = masker.mask(user_message, session_id)
    # masked_text goes to LLM
    # mapping stores {placeholder: original} for later use
"""

from __future__ import annotations

import re
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger("agentic.pii_masker")


@dataclass
class PIIPattern:
    """Definition of a PII pattern with its regex and placeholder prefix."""
    name: str
    pattern: str
    placeholder_prefix: str
    priority: int = 0  # Higher priority patterns are matched first


# PII patterns ordered by specificity (more specific patterns first)
PII_PATTERNS: List[PIIPattern] = [
    # Credit card - 13 to 19 digits with optional space/hyphen separators
    # Covers: Visa (13/16/19), MasterCard (16), Amex (15), Diners Club (14),
    #         Discover (16), JCB (15/16), UnionPay (16-19), Rupay (16)
    PIIPattern(
        name="credit_card",
        pattern=r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,4}(?:[\s\-]?\d{1,3})?\b",
        placeholder_prefix="CARD",
        priority=100,
    ),
    # NRIC/FIN - Singapore ID format (S/T/F/G/M + 7 digits + letter)
    PIIPattern(
        name="nric",
        pattern=r"\b[STFGM]\d{7}[A-Z]\b",
        placeholder_prefix="NRIC",
        priority=90,
    ),
    # Policy number - BigTapp format (2 letters + 6 digits)
    PIIPattern(
        name="policy_no",
        pattern=r"\b[A-Z]{2}\d{6}\b",
        placeholder_prefix="POLICY",
        priority=80,
    ),
    # Passport - common format (letter + 7-8 digits + optional letter)
    PIIPattern(
        name="passport",
        pattern=r"\b[A-Z]\d{7,8}[A-Z]?\b",
        placeholder_prefix="PASSPORT",
        priority=70,
    ),
    # Email - standard email format
    PIIPattern(
        name="email",
        pattern=r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
        placeholder_prefix="EMAIL",
        priority=60,
    ),
    # Mobile - Singapore format (+65 optional, starts with 6/8/9)
    PIIPattern(
        name="mobile",
        # Accept common separators (spaces/hyphens) e.g. "+65 9123 4567", "9123-4567"
        # IMPORTANT: Prevent matching as a substring inside longer digit sequences.
        # Example: "9894678650" should NOT be partially masked as "94678650".
        # Also accept '65' prefix without '+' (common user input): "65 9123 4567", "6591234567".
        pattern=r"(?<!\d)(?:\+?65[\s\-]?)?[689]\d{3}[\s\-]?\d{4}(?!\d)",
        placeholder_prefix="MOBILE",
        priority=50,
    ),
    # Postal code - Singapore 6 digits (with word boundaries to avoid matching other numbers)
    PIIPattern(
        name="postal_code",
        pattern=r"\b\d{6}\b",
        placeholder_prefix="POSTAL",
        priority=10,  # Lowest priority - matches many things
    ),
]


class PIIMasker:
    """
    Thread-safe PII masker with session-scoped mappings.
    
    Features:
    - Masks PII in user messages before LLM processing
    - Maintains session-scoped mappings for unmasking
    - Thread-safe for concurrent sessions
    - Debug logging for PII detection (at DEBUG level only)
    """
    
    def __init__(self):
        self._session_mappings: Dict[str, Dict[str, str]] = {}
        self._session_counters: Dict[str, Dict[str, int]] = {}
        self._lock = Lock()
        
        # Pre-compile patterns for performance
        # Compile with IGNORECASE for patterns that commonly appear in lowercase
        # (NRIC / policy / passport) so we don't accidentally leak PII to the LLM.
        ignorecase_names = {"email", "nric", "policy_no", "passport"}
        self._compiled_patterns = []
        for p in sorted(PII_PATTERNS, key=lambda x: -x.priority):
            flags = re.IGNORECASE if p.name in ignorecase_names else 0
            self._compiled_patterns.append((p, re.compile(p.pattern, flags)))
        
        logger.info("PIIMasker initialized with %d patterns", len(PII_PATTERNS))
    
    def mask(
        self,
        text: str,
        session_id: str,
        *,
        mask_postal_code: bool = True,
    ) -> Tuple[str, Dict[str, str]]:
        """
        Mask all PII in the given text.
        
        Args:
            text: User message that may contain PII
            session_id: Session identifier for tracking mappings
            
        Returns:
            Tuple of (masked_text, new_mappings)
            - masked_text: Text with PII replaced by placeholders
            - new_mappings: Dict of {placeholder: original} for this call
        """
        if not text:
            return text, {}
        
        with self._lock:
            # Initialize session data if needed
            if session_id not in self._session_mappings:
                self._session_mappings[session_id] = {}
                self._session_counters[session_id] = {}
            
            session_mapping = self._session_mappings[session_id]
            session_counters = self._session_counters[session_id]
        
        masked_text = text
        new_mappings: Dict[str, str] = {}
        
        # Track positions already masked to avoid double-masking
        masked_positions: List[Tuple[int, int]] = []
        
        for pii_pattern, compiled_re in self._compiled_patterns:
            if pii_pattern.name == "postal_code" and not mask_postal_code:
                continue
            # Find all matches
            for match in compiled_re.finditer(text):
                start, end = match.start(), match.end()
                original_value = match.group()
                
                # Skip if this position is already masked
                if any(start < e and end > s for s, e in masked_positions):
                    continue
                
                # ALWAYS create a new entry, even if the same value was seen before.
                # This ensures _get_latest_from_pii_mapping returns the most recent user input.
                # Example: User enters "123456" twice - we create [POSTAL_1] and [POSTAL_5]
                # so the latest lookup correctly returns the second instance.
                with self._lock:
                    counter = session_counters.get(pii_pattern.placeholder_prefix, 0) + 1
                    session_counters[pii_pattern.placeholder_prefix] = counter
                
                placeholder = f"[{pii_pattern.placeholder_prefix}_{counter}]"
                
                with self._lock:
                    session_mapping[placeholder] = original_value
                
                new_mappings[placeholder] = original_value
                
                masked_positions.append((start, end))
                
                logger.debug(
                    "PII.masked: type=%s placeholder=%s session=%s",
                    pii_pattern.name, placeholder, session_id
                )
        
        # Replace all matches using position-based replacement (reverse order to preserve offsets)
        sorted_positions = sorted(masked_positions, key=lambda x: x[0], reverse=True)
        position_to_placeholder: Dict[Tuple[int, int], str] = {}
        for (start, end), (placeholder, _) in zip(
            sorted(masked_positions, key=lambda x: x[0]),
            new_mappings.items()
        ):
            position_to_placeholder[(start, end)] = placeholder

        for (start, end) in sorted_positions:
            placeholder = position_to_placeholder.get((start, end))
            if placeholder:
                masked_text = masked_text[:start] + placeholder + masked_text[end:]
        
        if new_mappings:
            logger.info(
                "PII.mask_summary: session=%s detected=%d types=%s",
                session_id,
                len(new_mappings),
                list(set(p.split('_')[0].strip('[]') for p in new_mappings.keys()))
            )
        
        return masked_text, new_mappings
    
    def unmask(self, text: str, mapping: Optional[Dict[str, str]] = None) -> str:
        """
        Restore original PII values from placeholders.
        
        Args:
            text: Text containing placeholders
            mapping: Specific mapping to use (if None, returns text as-is)
            
        Returns:
            Text with placeholders replaced by original values
        """
        if not text or not mapping:
            return text
        
        result = text
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        
        return result
    
    def get_session_mapping(self, session_id: str) -> Dict[str, str]:
        """
        Get all PII mappings for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict of {placeholder: original} for the entire session
        """
        with self._lock:
            return dict(self._session_mappings.get(session_id, {}))
    
    def get_original_value(self, placeholder: str, session_id: str) -> Optional[str]:
        """
        Get the original value for a specific placeholder.
        
        Args:
            placeholder: The placeholder string (e.g., "[NRIC_1]")
            session_id: Session identifier
            
        Returns:
            Original PII value or None if not found
        """
        with self._lock:
            mapping = self._session_mappings.get(session_id, {})
            return mapping.get(placeholder)
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear all PII mappings for a session.
        
        Call this when a session ends to free memory.
        
        Args:
            session_id: Session identifier
        """
        with self._lock:
            self._session_mappings.pop(session_id, None)
            self._session_counters.pop(session_id, None)
        
        logger.debug("PII.session_cleared: session=%s", session_id)
    
    def extract_pii_by_type(self, session_id: str, pii_type: str) -> List[str]:
        """
        Extract all PII values of a specific type from session mapping.
        
        Args:
            session_id: Session identifier
            pii_type: Type prefix (e.g., "NRIC", "MOBILE", "EMAIL")
            
        Returns:
            List of original values for that PII type
        """
        with self._lock:
            mapping = self._session_mappings.get(session_id, {})
            return [
                orig for placeholder, orig in mapping.items()
                if placeholder.startswith(f"[{pii_type}_")
            ]


# Singleton instance
_pii_masker: Optional[PIIMasker] = None
_masker_lock = Lock()


def get_pii_masker() -> PIIMasker:
    """Get the singleton PII masker instance."""
    global _pii_masker
    if _pii_masker is None:
        with _masker_lock:
            if _pii_masker is None:
                _pii_masker = PIIMasker()
    return _pii_masker


def mask_pii(text: str, session_id: str) -> Tuple[str, Dict[str, str]]:
    """
    Convenience function to mask PII in text.
    
    Args:
        text: Text that may contain PII
        session_id: Session identifier
        
    Returns:
        Tuple of (masked_text, new_mappings)
    """
    return get_pii_masker().mask(text, session_id)


def unmask_pii(text: str, mapping: Dict[str, str]) -> str:
    """
    Convenience function to unmask PII in text.
    
    Args:
        text: Text containing placeholders
        mapping: Mapping of placeholders to original values
        
    Returns:
        Text with original values restored
    """
    return get_pii_masker().unmask(text, mapping)
