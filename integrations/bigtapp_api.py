"""
BigTapp Mock API Client for Policy Service Operations (Demo Mode)
=================================================================

MongoDB-backed mock client with LLM fallback. Demo customer data is
pre-seeded into MongoDB (via scripts/seed_demo_data.py) and persists
across server restarts.

Data flow:
1. On first access, loads ALL demo customers/claims from MongoDB into
   an in-memory cache for fast lookups.
2. validate_customer() checks cache first (name match), then MongoDB
   (NRIC match), then falls back to LLM generation.
3. LLM-generated customers are written back to MongoDB so they persist.
4. Updates modify both in-memory cache AND MongoDB.

Collections used:
    demo_customers  — customer profiles with embedded policies
    demo_claims     — claim records linked by NRIC and policyNo

API Endpoints (mocked):
- POST /api/v1/customer/validate
- GET /api/v1/claim/list/{nric}
- GET /api/v1/chatbot/policies?nric={nric}
- GET /api/v1/policies/{policyNo}
- POST /api/v1/customer/update
- POST /api/v1/home-protect-insured-address/update
- GET /api/v1/postalCode/{postalCode}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentic.service_flow")

BIGTAPP_API_BASE_URL = os.getenv("BIGTAPP_API_BASE_URL", "mock://demo")
BIGTAPP_API_TIMEOUT = float(os.getenv("BIGTAPP_API_TIMEOUT", "30.0"))
BIGTAPP_API_UPDATE_TIMEOUT = float(os.getenv("BIGTAPP_API_UPDATE_TIMEOUT", "60.0"))
BIGTAPP_API_LOG_PII = os.getenv("BIGTAPP_API_LOG_PII", "false").strip().lower() in (
    "1", "true", "yes", "y",
)


class UpdateType(str, Enum):
    MOBILE_CHANGE = "mobile_change"
    EMAIL_CHANGE = "email_change"
    ADDRESS_CHANGE = "address_change"
    PAYMENT_INFO_CHANGE = "payment_info_change"
    FULL_INFO_CHANGE = "full_info_change"


@dataclass
class APIError:
    code: int
    message: str
    details: Optional[str] = None


_NRIC_RE = re.compile(r"\b[STFG]\d{7}[A-Z]\b", re.IGNORECASE)
_POLICY_RE = re.compile(r"\b[A-Z]{2}\d{6}\b", re.IGNORECASE)
_MOBILE_RE = re.compile(r"(?<!\d)(?:\+?65[\s\-]?)?[689]\d{3}[\s\-]?\d{4}(?!\d)")
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_POSTAL_RE = re.compile(r"\b\d{6}\b")


def _mask_text_for_log(text: str) -> str:
    if not text:
        return text
    t = str(text)
    t = _EMAIL_RE.sub("[EMAIL]", t)
    t = _NRIC_RE.sub("[NRIC]", t)
    t = _POLICY_RE.sub("[POLICY]", t)
    t = _MOBILE_RE.sub("[MOBILE]", t)
    t = _POSTAL_RE.sub("[POSTAL]", t)
    return t


def _safe_json_for_log(obj: Any, max_len: int = 1200) -> str:
    try:
        raw = json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        raw = str(obj)
    text = raw if BIGTAPP_API_LOG_PII else _mask_text_for_log(raw)
    if max_len and len(text) > max_len:
        return text[:max_len] + "... [TRUNCATED]"
    return text


def _safe_json_for_info(obj: Any, max_len: int = 1200) -> str:
    try:
        raw = json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        raw = str(obj)
    text = _mask_text_for_log(raw)
    if max_len and len(text) > max_len:
        return text[:max_len] + "... [TRUNCATED]"
    return text


# =========================================================================
# LLM DATA GENERATION
# =========================================================================

_CUSTOMER_GENERATION_PROMPT = """You are a data generator for a demo insurance chatbot. Generate a realistic
customer profile for a Singapore insurance customer.

Customer details provided:
- First name: {first_name}
- Last name: {last_name}
- Email: {email}
- Mobile: {mobile}

Generate a JSON object with this EXACT structure (no markdown, just raw JSON):
{{
  "idCardNumber": "<Singapore NRIC format: S followed by 7 digits and a letter, e.g. S1234567A>",
  "givenName": "{first_name}",
  "surname": "{last_name}",
  "emailAddress": "{email}",
  "mobileNo": "{mobile}",
  "dateOfBirth": "<ISO date, age 25-55>",
  "gender": "<Male or Female>",
  "address": {{
    "postalCode": "<6-digit Singapore postal code>",
    "unitNo": "<e.g. #12-345>",
    "blockHouseNumber": "<e.g. 123>",
    "streetName": "<realistic Singapore street name>",
    "buildingName": "<realistic Singapore condo/HDB name or empty string>"
  }},
  "policies": [
    {{
      "policyNo": "<2 uppercase letters + 6 digits, e.g. TA300123>",
      "productName": "<one of: Travel Protect360, Family Protect360, Home Protect360, Car Protect360, Maid Protect360 PRO, Early Protect360 Plus, Fraud Protect360 Plus, Hospital Protect360, ChoiceProtect360>",
      "status": "<Active or Lapsed>",
      "commencementDate": "<ISO date within last 2 years>",
      "policyEndDate": "<ISO date, 1 year after commencement>",
      "premiumAmount": <number between 15 and 200>,
      "paymentFrequency": "<Monthly or Annual>"
    }}
  ]
}}

RULES:
- Generate 2-4 policies with realistic mix of products
- Use correct policy prefixes: TA/TB=Travel, FA=Family, HC=Home, MP=Car, DY=Maid, ES=Early, CY=Fraud, HI=Hospital, CK=Choice
- Most policies should be Active, maybe 1 Lapsed
- Return ONLY the JSON, no explanation"""

_CLAIMS_GENERATION_PROMPT = """You are a data generator for a demo insurance chatbot. Generate realistic
insurance claims for a customer.

Customer NRIC: {nric}
Customer policies: {policies_summary}

Generate a JSON array of 1-3 claims with this EXACT structure (no markdown, just raw JSON):
[
  {{
    "claimNo": "<CLM followed by 6 digits, e.g. CLM100234>",
    "policyNo": "<must match one of the customer's actual policy numbers>",
    "productName": "<product name matching the policy>",
    "status": "<one of: Processing, Approved, Rejected>",
    "claimAmount": <number between 200 and 15000>,
    "claimDate": "<ISO date within last 6 months>",
    "claimType": "<relevant to the product, e.g. Medical Expense, Trip Cancellation, Accidental Damage>",
    "description": "<brief 1-sentence description>"
  }}
]

RULES:
- At least 1 claim should be "Processing" (pending)
- Claims must reference actual policy numbers from the customer's policies
- Return ONLY the JSON array, no explanation"""


async def _generate_with_llm(prompt: str) -> Optional[str]:
    """Call the router LLM to generate mock data."""
    try:
        from ..infrastructure import get_router_llm
        llm = get_router_llm()
        response = await llm.ainvoke(prompt)
        content = str(getattr(response, "content", "") or "").strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()
        return content
    except Exception as e:
        logger.error("MockAPI.llm_generate.failed: %s", e)
        return None


def _parse_json_response(text: Optional[str]) -> Any:
    """Parse JSON from LLM response, handling common formatting issues."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
        logger.warning("MockAPI.parse_json.failed: could not parse LLM output")
        return None


# =========================================================================
# FALLBACK DATA (used if LLM generation fails)
# =========================================================================

def _generate_fallback_customer(first_name: str, last_name: str, email: str, mobile: str) -> Dict:
    """Generate deterministic fallback customer data without LLM."""
    nric_num = int(hashlib.sha256(f"{first_name}{last_name}".encode()).hexdigest(), 16) % 9000000 + 1000000
    nric = f"S{nric_num}A"

    now = datetime.now()
    policies = [
        {
            "policyNo": f"TA{300000 + abs(hash(first_name)) % 999:06d}",
            "productName": "Travel Protect360",
            "status": "Active",
            "commencementDate": (now - timedelta(days=120)).strftime("%Y-%m-%dT00:00:00"),
            "policyEndDate": (now + timedelta(days=245)).strftime("%Y-%m-%dT00:00:00"),
            "premiumAmount": 45.90,
            "paymentFrequency": "Annual",
        },
        {
            "policyNo": f"HC{300000 + abs(hash(last_name)) % 999:06d}",
            "productName": "Home Protect360",
            "status": "Active",
            "commencementDate": (now - timedelta(days=300)).strftime("%Y-%m-%dT00:00:00"),
            "policyEndDate": (now + timedelta(days=65)).strftime("%Y-%m-%dT00:00:00"),
            "premiumAmount": 38.89,
            "paymentFrequency": "Monthly",
        },
        {
            "policyNo": f"FA{300000 + abs(hash(email)) % 999:06d}",
            "productName": "Family Protect360",
            "status": "Lapsed",
            "commencementDate": (now - timedelta(days=500)).strftime("%Y-%m-%dT00:00:00"),
            "policyEndDate": (now - timedelta(days=135)).strftime("%Y-%m-%dT00:00:00"),
            "premiumAmount": 29.50,
            "paymentFrequency": "Monthly",
        },
    ]

    return {
        "idCardNumber": nric,
        "givenName": first_name,
        "surname": last_name,
        "emailAddress": email,
        "mobileNo": mobile,
        "dateOfBirth": "1988-06-15T00:00:00",
        "gender": "Male",
        "address": {
            "postalCode": "520123",
            "unitNo": "#08-456",
            "blockHouseNumber": "123",
            "streetName": "Ang Mo Kio Avenue 6",
            "buildingName": "",
        },
        "policies": policies,
    }


def _generate_fallback_claims(nric: str, policies: List[Dict]) -> List[Dict]:
    """Generate deterministic fallback claims without LLM."""
    if not policies:
        return []

    now = datetime.now()
    active_policies = [p for p in policies if p.get("status", "").lower() == "active"]
    target = active_policies[0] if active_policies else policies[0]

    claim_types = {
        "Travel Protect360": ("Medical Expense", "Emergency medical treatment during overseas trip"),
        "Home Protect360": ("Property Damage", "Water damage to living room flooring"),
        "Family Protect360": ("Accident Medical", "Outpatient treatment after minor fall"),
        "Car Protect360": ("Vehicle Repair", "Windscreen crack from road debris"),
        "Maid Protect360 PRO": ("Medical Expense", "Helper hospitalization for illness"),
        "Early Protect360 Plus": ("Critical Illness", "Early-stage critical illness claim"),
        "Fraud Protect360 Plus": ("Online Fraud", "Unauthorized online transaction"),
        "Hospital Protect360": ("Hospital Cash", "3-day hospitalization"),
        "ChoiceProtect360": ("Accident Medical", "Accident medical reimbursement"),
    }

    product = target.get("productName", "Travel Protect360")
    ct, desc = claim_types.get(product, ("General", "Insurance claim"))

    return [
        {
            "claimNo": f"CLM{100000 + int(hashlib.sha256(nric.encode()).hexdigest(), 16) % 9999:06d}",
            "policyNo": target.get("policyNo", "TA300001"),
            "productName": product,
            "status": "Processing",
            "claimAmount": round(1500 + (int(hashlib.sha256(nric.encode()).hexdigest(), 16) % 3500), 2),
            "claimDate": (now - timedelta(days=10 + int(hashlib.sha256(nric.encode()).hexdigest(), 16) % 35)).strftime("%Y-%m-%dT00:00:00"),
            "claimType": ct,
            "description": desc,
        },
    ]


# =========================================================================
# MONGODB-BACKED CACHE
# =========================================================================

_customer_cache: Dict[str, Dict[str, Any]] = {}
_claims_cache: Dict[str, List[Dict]] = {}
_mongo_loaded: bool = False


_mongo_client = None
_mongo_db_handle = None

def _get_mongo_db():
    """Get the MongoDB database handle (lazy, reuses pymongo connection)."""
    global _mongo_client, _mongo_db_handle
    if _mongo_db_handle is not None:
        return _mongo_db_handle
    try:
        from pymongo import MongoClient
        mongo_uri = os.getenv("MONGO_URI")
        db_name = os.getenv("DB_NAME", "bigtapp").lower()
        if not mongo_uri:
            return None
        _mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        _mongo_client.admin.command("ping")
        _mongo_db_handle = _mongo_client[db_name]
        return _mongo_db_handle
    except Exception as e:
        logger.warning("MOCK_API: MongoDB unavailable, using in-memory only: %s", e)
        return None


def _load_demo_data_from_mongo() -> None:
    """Load pre-seeded demo customers and claims from MongoDB into cache."""
    global _mongo_loaded
    if _mongo_loaded:
        return

    _mongo_loaded = True
    db = _get_mongo_db()
    if db is None:
        logger.info("MOCK_API: No MongoDB — starting with empty cache (LLM fallback)")
        return

    try:
        cust_col = db["demo_customers"]
        claim_col = db["demo_claims"]

        loaded_customers = 0
        for doc in cust_col.find({}, {"_id": 0, "_seeded_at": 0}):
            name_key = f"{(doc.get('givenName') or '').strip().lower()}_{(doc.get('surname') or '').strip().lower()}"
            _customer_cache[name_key] = doc
            loaded_customers += 1

        loaded_claims = 0
        for doc in claim_col.find({}, {"_id": 0, "_seeded_at": 0}):
            nric = doc.get("nric", "")
            if nric not in _claims_cache:
                _claims_cache[nric] = []
            _claims_cache[nric].append(doc)
            loaded_claims += 1

        logger.info(
            "MOCK_API: Loaded %d customers and %d claims from MongoDB",
            loaded_customers, loaded_claims,
        )
    except Exception as e:
        logger.warning("MOCK_API: Failed to load demo data from MongoDB: %s", e)


def _persist_customer_to_mongo(customer_data: Dict[str, Any]) -> None:
    """Write a customer record back to MongoDB for persistence."""
    try:
        db = _get_mongo_db()
        if db is None:
            return
        nric = customer_data.get("idCardNumber", "")
        if not nric:
            return
        db["demo_customers"].replace_one(
            {"idCardNumber": nric}, customer_data, upsert=True,
        )
    except Exception as e:
        logger.debug("MOCK_API: Could not persist customer to MongoDB: %s", e)


def _persist_claims_to_mongo(nric: str, claims: List[Dict]) -> None:
    """Write claim records back to MongoDB for persistence."""
    try:
        db = _get_mongo_db()
        if db is None:
            return
        col = db["demo_claims"]
        for claim in claims:
            claim_no = claim.get("claimNo", "")
            if claim_no:
                col.replace_one({"claimNo": claim_no}, {**claim, "nric": nric}, upsert=True)
    except Exception as e:
        logger.debug("MOCK_API: Could not persist claims to MongoDB: %s", e)


# =========================================================================
# MOCK API CLIENT
# =========================================================================


class BigTappApiClient:
    """
    MongoDB-backed mock API client for BigTapp demo.

    Same interface as the original HTTP client so the service subgraph
    needs zero changes. Demo data is loaded from MongoDB on first access;
    LLM generation is the fallback for unknown customers.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = base_url or BIGTAPP_API_BASE_URL
        self.timeout = timeout or BIGTAPP_API_TIMEOUT
        _load_demo_data_from_mongo()
        logger.info("BIGTAPP_CLIENT_INIT: Mock API client created (MongoDB + LLM fallback)")

    async def close(self) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    # =====================================================================
    # Customer Validation
    # =====================================================================

    async def validate_customer(
        self,
        first_name: str,
        last_name: str,
        email: str,
        mobile: str,
    ) -> Dict[str, Any]:
        req_id = uuid.uuid4().hex[:10]
        t0 = time.perf_counter()
        cache_key = f"{(first_name or '').strip().lower()}_{(last_name or '').strip().lower()}"

        logger.info(
            "MOCK_API: validate_customer id=%s name=%s %s",
            req_id, (first_name or "")[:2], (last_name or "")[:2],
        )

        # Return cached data if available
        if cache_key in _customer_cache:
            cached = _customer_cache[cache_key]
            # Update mutable fields from current request
            cached["emailAddress"] = email
            cached["mobileNo"] = mobile
            logger.info("MOCK_API: validate_customer cache_hit id=%s duration=%.3fs",
                        req_id, time.perf_counter() - t0)
            return {"success": True, "data": cached, "request_id": req_id}

        # Customer not in our records - block access (negative scenario handling)
        logger.info(
            "MOCK_API: validate_customer not_found id=%s cache_size=%d duration=%.3fs",
            req_id, len(_customer_cache), time.perf_counter() - t0,
        )
        return {
            "success": False,
            "error": "We couldn't find your details in our records. Please double-check your first name, last name, email, and mobile number, or contact BigTapp support if you need help.",
            "status_code": 404,
            "request_id": req_id,
        }

        # Generate via LLM (unreachable - kept for reference)
        prompt = _CUSTOMER_GENERATION_PROMPT.format(
            first_name=first_name or "John",
            last_name=last_name or "Tan",
            email=email or "john.tan@example.com",
            mobile=mobile or "91234567",
        )

        raw = await _generate_with_llm(prompt)
        customer_data = _parse_json_response(raw)

        if not customer_data or not isinstance(customer_data, dict):
            logger.warning("MOCK_API: LLM generation failed, using fallback data")
            customer_data = _generate_fallback_customer(
                first_name or "John",
                last_name or "Tan",
                email or "john.tan@example.com",
                mobile or "91234567",
            )

        # Ensure critical fields match input
        customer_data["givenName"] = first_name or customer_data.get("givenName", "John")
        customer_data["surname"] = last_name or customer_data.get("surname", "Tan")
        customer_data["emailAddress"] = email or customer_data.get("emailAddress", "")
        customer_data["mobileNo"] = mobile or customer_data.get("mobileNo", "")

        # Ensure NRIC exists
        if not customer_data.get("idCardNumber"):
            nric_num = abs(hash(cache_key)) % 9000000 + 1000000
            customer_data["idCardNumber"] = f"S{nric_num}A"

        # Ensure policies exist
        if not customer_data.get("policies") or not isinstance(customer_data["policies"], list):
            customer_data["policies"] = _generate_fallback_customer(
                first_name, last_name, email, mobile
            )["policies"]

        # Validate policy format
        for p in customer_data.get("policies", []):
            if not p.get("policyNo") or not re.match(r"^[A-Z]{2}\d{6}$", str(p["policyNo"])):
                prefix = random.choice(["TA", "HC", "FA", "MP", "DY", "ES", "CY", "HI", "CK"])
                p["policyNo"] = f"{prefix}{random.randint(300000, 399999)}"

        _customer_cache[cache_key] = customer_data
        _persist_customer_to_mongo(customer_data)

        duration = time.perf_counter() - t0
        logger.info("MOCK_API: validate_customer success id=%s policies=%d duration=%.3fs",
                     req_id, len(customer_data.get("policies", [])), duration)
        return {"success": True, "data": customer_data, "request_id": req_id}

    # =====================================================================
    # Claims
    # =====================================================================

    async def get_claims(self, nric: str) -> Dict[str, Any]:
        req_id = uuid.uuid4().hex[:10]
        t0 = time.perf_counter()

        logger.info("MOCK_API: get_claims id=%s nric=***%s", req_id, (nric or "")[-4:])

        # Return cached claims if available
        if nric in _claims_cache:
            logger.info("MOCK_API: get_claims cache_hit id=%s", req_id)
            return {"success": True, "data": _claims_cache[nric], "request_id": req_id}

        # Find customer data for this NRIC
        customer_data = None
        for cd in _customer_cache.values():
            if cd.get("idCardNumber") == nric:
                customer_data = cd
                break

        policies = customer_data.get("policies", []) if customer_data else []
        policies_summary = ", ".join(
            f"{p.get('policyNo')} ({p.get('productName')})" for p in policies
        ) if policies else "TA300001 (Travel Protect360)"

        prompt = _CLAIMS_GENERATION_PROMPT.format(
            nric=nric,
            policies_summary=policies_summary,
        )

        raw = await _generate_with_llm(prompt)
        claims = _parse_json_response(raw)

        if not claims or not isinstance(claims, list):
            logger.warning("MOCK_API: Claims LLM generation failed, using fallback")
            claims = _generate_fallback_claims(nric, policies)

        # Validate claim structure
        for c in claims:
            if not c.get("claimNo"):
                c["claimNo"] = f"CLM{random.randint(100000, 199999)}"
            if not c.get("status"):
                c["status"] = "Processing"

        _claims_cache[nric] = claims
        _persist_claims_to_mongo(nric, claims)

        duration = time.perf_counter() - t0
        logger.info("MOCK_API: get_claims success id=%s claims=%d duration=%.3fs",
                     req_id, len(claims), duration)
        return {"success": True, "data": claims, "request_id": req_id}

    # =====================================================================
    # Policies
    # =====================================================================

    async def get_policies(self, nric: str) -> Dict[str, Any]:
        req_id = uuid.uuid4().hex[:10]
        logger.info("MOCK_API: get_policies id=%s nric=***%s", req_id, (nric or "")[-4:])

        # Find customer data for this NRIC
        for cd in _customer_cache.values():
            if cd.get("idCardNumber") == nric:
                policies = cd.get("policies", [])
                return {"success": True, "data": policies, "request_id": req_id}

        # No cached customer — return empty
        return {"success": True, "data": [], "request_id": req_id}

    async def get_policy_details(self, policy_no: str) -> Dict[str, Any]:
        req_id = uuid.uuid4().hex[:10]
        logger.info("MOCK_API: get_policy_details id=%s policy=%s",
                     req_id, f"{(policy_no or '')[:2]}...{(policy_no or '')[-2:]}")

        # Search all cached customers for this policy
        for cd in _customer_cache.values():
            for p in cd.get("policies", []):
                if p.get("policyNo") == policy_no:
                    detail = {
                        **p,
                        "customerName": f"{cd.get('givenName', '')} {cd.get('surname', '')}".strip(),
                        "nric": cd.get("idCardNumber", ""),
                        "email": cd.get("emailAddress", ""),
                        "mobile": cd.get("mobileNo", ""),
                    }
                    return {"success": True, "data": detail, "request_id": req_id}

        return {
            "success": False,
            "error": f"Policy {policy_no} not found.",
            "status_code": 404,
            "request_id": req_id,
        }

    # =====================================================================
    # Customer Updates (all return immediate success for demo)
    # =====================================================================

    async def update_customer(
        self,
        nric: str,
        update_type: UpdateType,
        update_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        req_id = uuid.uuid4().hex[:10]
        logger.info("MOCK_API: update_customer id=%s type=%s nric=***%s",
                     req_id, update_type.value, (nric or "")[-4:])

        # Update cached customer data and persist
        for cd in _customer_cache.values():
            if cd.get("idCardNumber") == nric:
                if update_type == UpdateType.EMAIL_CHANGE and "email" in update_data:
                    cd["emailAddress"] = update_data["email"]
                elif update_type == UpdateType.MOBILE_CHANGE and "mobile" in update_data:
                    cd["mobileNo"] = update_data["mobile"]
                elif update_type == UpdateType.ADDRESS_CHANGE:
                    cd["address"] = {
                        "postalCode": update_data.get("postalCode", ""),
                        "unitNo": update_data.get("unitNo", ""),
                        "blockHouseNumber": update_data.get("houseNo", ""),
                        "streetName": update_data.get("streetName", ""),
                        "buildingName": update_data.get("buildingName", ""),
                    }
                _persist_customer_to_mongo(cd)
                return {"success": True, "data": cd, "request_id": req_id}

        return {"success": True, "data": {}, "request_id": req_id}

    async def update_email(self, nric: str, new_email: str) -> Dict[str, Any]:
        return await self.update_customer(
            nric=nric,
            update_type=UpdateType.EMAIL_CHANGE,
            update_data={"email": new_email},
        )

    async def update_mobile(self, nric: str, new_mobile: str) -> Dict[str, Any]:
        return await self.update_customer(
            nric=nric,
            update_type=UpdateType.MOBILE_CHANGE,
            update_data={"mobile": new_mobile},
        )

    async def update_address(
        self,
        nric: str,
        postal_code: str,
        unit_no: str,
        house_no: str,
        street_name: str,
        building_name: str = "",
    ) -> Dict[str, Any]:
        return await self.update_customer(
            nric=nric,
            update_type=UpdateType.ADDRESS_CHANGE,
            update_data={
                "postalCode": postal_code,
                "unitNo": unit_no,
                "houseNo": house_no,
                "streetName": street_name,
                "buildingName": building_name,
            },
        )

    async def update_payment_info(
        self,
        nric: str,
        card_no: str,
        card_expire: str,
        credit_card_type: str,
        policy_no: str,
        payer_surname: str,
        payer_given_name: str,
        payer_nric: str,
    ) -> Dict[str, Any]:
        req_id = uuid.uuid4().hex[:10]
        logger.info("MOCK_API: update_payment_info id=%s nric=***%s policy=%s",
                     req_id, (nric or "")[-4:], (policy_no or "")[:2])
        return {"success": True, "data": {"status": "updated"}, "request_id": req_id}

    # =====================================================================
    # Home Protect Insured Address
    # =====================================================================

    async def update_insured_address(
        self,
        policy_no: str,
        postal_code: str,
        unit_no: str,
        house_no: str,
        street_name: str,
        building_name: str = "",
    ) -> Dict[str, Any]:
        req_id = uuid.uuid4().hex[:10]
        logger.info("MOCK_API: update_insured_address id=%s policy=%s",
                     req_id, f"{(policy_no or '')[:2]}...{(policy_no or '')[-2:]}")
        return {"success": True, "data": {"status": "updated"}, "request_id": req_id}

    # =====================================================================
    # Utilities
    # =====================================================================

    async def get_postal_code_info(self, postal_code: str) -> Dict[str, Any]:
        req_id = uuid.uuid4().hex[:10]
        logger.info("MOCK_API: get_postal_code_info id=%s postal=****%s",
                     req_id, (postal_code or "")[-2:])

        # Return a realistic Singapore address based on the postal code prefix
        prefix = (postal_code or "000000")[:2]
        address_map = {
            "01": ("1", "Raffles Place", "One Raffles Place"),
            "02": ("10", "Collyer Quay", "Ocean Financial Centre"),
            "03": ("1", "Temasek Avenue", "Millenia Tower"),
            "04": ("1", "Harbourfront Walk", "VivoCity"),
            "05": ("3", "Bukit Merah Central", "ABC Building"),
            "10": ("20", "Bukit Timah Road", ""),
            "11": ("50", "Orchard Road", "ION Orchard"),
            "12": ("30", "Orange Grove Road", ""),
            "13": ("10", "Toa Payoh Lorong 1", ""),
            "14": ("25", "Geylang Road", ""),
            "15": ("20", "Marine Parade Road", ""),
            "16": ("10", "Bedok North Street 3", ""),
            "17": ("15", "Changi Business Park", "UE BizHub East"),
            "18": ("50", "Tampines Avenue 5", "Our Tampines Hub"),
            "19": ("20", "Punggol Central", "Waterway Point"),
            "20": ("10", "Bishan Street 11", ""),
            "22": ("30", "Boon Lay Way", "Jurong Point"),
            "23": ("60", "Jurong Gateway Road", "JEM"),
            "24": ("10", "Lim Chu Kang Road", ""),
            "25": ("20", "Woodlands Avenue 1", ""),
            "26": ("10", "Mandai Lake Road", ""),
            "27": ("20", "Yishun Avenue 2", "Northpoint City"),
            "28": ("10", "Seletar Aerospace Drive", ""),
            "31": ("5", "Lower Kent Ridge Road", "NUS"),
            "34": ("30", "Geylang East Avenue 1", ""),
            "38": ("15", "Simei Street 4", "Eastpoint Mall"),
            "40": ("1", "Pasir Ris Close", ""),
            "46": ("20", "Bukit Batok West Avenue 6", ""),
            "47": ("5", "Clementi Avenue 2", ""),
            "48": ("10", "Choa Chu Kang Avenue 4", ""),
            "50": ("20", "Serangoon Avenue 3", "NEX"),
            "51": ("30", "Ang Mo Kio Avenue 3", "AMK Hub"),
            "52": ("123", "Ang Mo Kio Avenue 6", ""),
            "53": ("20", "Hougang Avenue 10", "Hougang Mall"),
            "54": ("10", "Sengkang Square", "Compass One"),
            "56": ("5", "Senoko Road", ""),
            "60": ("10", "Bukit Timah Road", ""),
            "65": ("20", "Lorong Chuan", ""),
            "68": ("15", "Upper Serangoon Road", ""),
            "69": ("10", "Hougang Street 21", ""),
            "72": ("20", "Ubi Avenue 1", ""),
            "73": ("5", "Toa Payoh Rise", ""),
            "75": ("10", "Stirling Road", ""),
            "79": ("20", "Toa Payoh North", ""),
            "80": ("30", "Bedok Reservoir Road", ""),
            "82": ("15", "Marine Parade Central", "Parkway Parade"),
        }

        block, street, building = address_map.get(prefix, ("10", "Orchard Road", ""))

        return {
            "success": True,
            "data": {
                "streetName": street,
                "buildingName": building,
                "blockHouseNumber": block,
            },
            "request_id": req_id,
        }


# =========================================================================
# Singleton
# =========================================================================

_api_client: Optional[BigTappApiClient] = None


def get_bigtapp_api_client() -> BigTappApiClient:
    global _api_client
    if _api_client is None:
        _api_client = BigTappApiClient()
    return _api_client


async def close_bigtapp_api_client() -> None:
    global _api_client
    if _api_client is not None:
        await _api_client.close()
        _api_client = None
