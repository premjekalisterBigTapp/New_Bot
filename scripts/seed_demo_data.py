#!/usr/bin/env python3
"""
Seed Demo Customer Data into MongoDB
=====================================

Pre-loads realistic synthetic customer profiles, policies, and claims
into MongoDB for consistent demo behavior. Data persists across server
restarts via the bigtapp-mongodb Docker volume.

Collections created:
    - demo_customers: Customer profiles with embedded policies
    - demo_claims:    Insurance claim records linked to policies

Usage:
    python scripts/seed_demo_data.py              # Seed (skip if exists)
    python scripts/seed_demo_data.py --reset      # Drop & re-seed
    python scripts/seed_demo_data.py --show       # Print current data
"""

from __future__ import annotations

import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from pymongo import MongoClient, ASCENDING
except ImportError:
    logger.error("pymongo is required. Install with: pip install pymongo")
    sys.exit(1)


# =========================================================================
# DEMO DATA — 5 realistic Singapore insurance customers
# =========================================================================

DEMO_CUSTOMERS = [
    {
        "idCardNumber": "S8834567A",
        "givenName": "John",
        "surname": "Tan",
        "emailAddress": "john.tan@gmail.com",
        "mobileNo": "91234567",
        "dateOfBirth": "1988-06-15T00:00:00",
        "gender": "Male",
        "address": {
            "postalCode": "520123",
            "unitNo": "#08-456",
            "blockHouseNumber": "123",
            "streetName": "Ang Mo Kio Avenue 6",
            "buildingName": "",
        },
        "policies": [
            {
                "policyNo": "TA300101",
                "productName": "Travel Protect360",
                "status": "Active",
                "commencementDate": "2025-09-15T00:00:00",
                "policyEndDate": "2026-09-14T00:00:00",
                "premiumAmount": 45.90,
                "paymentFrequency": "Annual",
            },
            {
                "policyNo": "HC300201",
                "productName": "Home Protect360",
                "status": "Active",
                "commencementDate": "2025-04-01T00:00:00",
                "policyEndDate": "2026-03-31T00:00:00",
                "premiumAmount": 38.89,
                "paymentFrequency": "Monthly",
            },
            {
                "policyNo": "FA300301",
                "productName": "Family Protect360",
                "status": "Lapsed",
                "commencementDate": "2024-06-01T00:00:00",
                "policyEndDate": "2025-05-31T00:00:00",
                "premiumAmount": 29.50,
                "paymentFrequency": "Monthly",
            },
        ],
    },
    {
        "idCardNumber": "S9245678B",
        "givenName": "Sarah",
        "surname": "Lim",
        "emailAddress": "sarah.lim@outlook.com",
        "mobileNo": "82345678",
        "dateOfBirth": "1992-03-22T00:00:00",
        "gender": "Female",
        "address": {
            "postalCode": "238801",
            "unitNo": "#15-02",
            "blockHouseNumber": "50",
            "streetName": "Orchard Road",
            "buildingName": "ION Residences",
        },
        "policies": [
            {
                "policyNo": "MP300401",
                "productName": "Car Protect360",
                "status": "Active",
                "commencementDate": "2025-07-10T00:00:00",
                "policyEndDate": "2026-07-09T00:00:00",
                "premiumAmount": 156.00,
                "paymentFrequency": "Monthly",
            },
            {
                "policyNo": "CY300501",
                "productName": "Fraud Protect360 Plus",
                "status": "Active",
                "commencementDate": "2025-11-01T00:00:00",
                "policyEndDate": "2026-10-31T00:00:00",
                "premiumAmount": 18.90,
                "paymentFrequency": "Annual",
            },
            {
                "policyNo": "HI300601",
                "productName": "Hospital Protect360",
                "status": "Active",
                "commencementDate": "2025-05-20T00:00:00",
                "policyEndDate": "2026-05-19T00:00:00",
                "premiumAmount": 42.00,
                "paymentFrequency": "Monthly",
            },
        ],
    },
    {
        "idCardNumber": "S7856789C",
        "givenName": "Raj",
        "surname": "Kumar",
        "emailAddress": "raj.kumar@yahoo.com",
        "mobileNo": "93456789",
        "dateOfBirth": "1978-11-08T00:00:00",
        "gender": "Male",
        "address": {
            "postalCode": "460020",
            "unitNo": "#04-123",
            "blockHouseNumber": "20",
            "streetName": "Bishan Street 11",
            "buildingName": "",
        },
        "policies": [
            {
                "policyNo": "ES300701",
                "productName": "Early Protect360 Plus",
                "status": "Active",
                "commencementDate": "2025-01-15T00:00:00",
                "policyEndDate": "2026-01-14T00:00:00",
                "premiumAmount": 68.50,
                "paymentFrequency": "Monthly",
            },
            {
                "policyNo": "DY300801",
                "productName": "Maid Protect360 PRO",
                "status": "Active",
                "commencementDate": "2025-08-01T00:00:00",
                "policyEndDate": "2026-07-31T00:00:00",
                "premiumAmount": 25.00,
                "paymentFrequency": "Annual",
            },
            {
                "policyNo": "CK300901",
                "productName": "ChoiceProtect360",
                "status": "Lapsed",
                "commencementDate": "2024-03-01T00:00:00",
                "policyEndDate": "2025-02-28T00:00:00",
                "premiumAmount": 15.90,
                "paymentFrequency": "Monthly",
            },
        ],
    },
    {
        "idCardNumber": "T0167890D",
        "givenName": "Mei Ling",
        "surname": "Wong",
        "emailAddress": "meiling.wong@gmail.com",
        "mobileNo": "84567890",
        "dateOfBirth": "1995-09-12T00:00:00",
        "gender": "Female",
        "address": {
            "postalCode": "821350",
            "unitNo": "#12-567",
            "blockHouseNumber": "350",
            "streetName": "Tampines Street 33",
            "buildingName": "",
        },
        "policies": [
            {
                "policyNo": "TB301001",
                "productName": "Travel Protect360",
                "status": "Active",
                "commencementDate": "2026-01-05T00:00:00",
                "policyEndDate": "2027-01-04T00:00:00",
                "premiumAmount": 52.00,
                "paymentFrequency": "Annual",
            },
            {
                "policyNo": "FA301101",
                "productName": "Family Protect360",
                "status": "Active",
                "commencementDate": "2025-10-15T00:00:00",
                "policyEndDate": "2026-10-14T00:00:00",
                "premiumAmount": 34.90,
                "paymentFrequency": "Monthly",
            },
        ],
    },
    {
        "idCardNumber": "S7078901E",
        "givenName": "David",
        "surname": "Chen",
        "emailAddress": "david.chen@hotmail.com",
        "mobileNo": "95678901",
        "dateOfBirth": "1970-04-30T00:00:00",
        "gender": "Male",
        "address": {
            "postalCode": "159363",
            "unitNo": "#03-08",
            "blockHouseNumber": "5",
            "streetName": "Lower Kent Ridge Road",
            "buildingName": "Kent Ridge Residences",
        },
        "policies": [
            {
                "policyNo": "HC301201",
                "productName": "Home Protect360",
                "status": "Active",
                "commencementDate": "2025-06-01T00:00:00",
                "policyEndDate": "2026-05-31T00:00:00",
                "premiumAmount": 45.00,
                "paymentFrequency": "Monthly",
            },
            {
                "policyNo": "MP301301",
                "productName": "Car Protect360",
                "status": "Active",
                "commencementDate": "2025-12-01T00:00:00",
                "policyEndDate": "2026-11-30T00:00:00",
                "premiumAmount": 189.00,
                "paymentFrequency": "Monthly",
            },
            {
                "policyNo": "HI301401",
                "productName": "Hospital Protect360",
                "status": "Active",
                "commencementDate": "2025-03-15T00:00:00",
                "policyEndDate": "2026-03-14T00:00:00",
                "premiumAmount": 55.00,
                "paymentFrequency": "Monthly",
            },
        ],
    },
    {
        "idCardNumber": "S9293940Z",
        "givenName": "Azhar",
        "surname": "S",
        "emailAddress": "Azhar@gmail.com",
        "mobileNo": "92939495",
        "dateOfBirth": "1995-01-01T00:00:00",
        "gender": "Male",
        "address": {
            "postalCode": "540221",
            "unitNo": "#07-88",
            "blockHouseNumber": "221",
            "streetName": "Yishun Avenue 5",
            "buildingName": "",
        },
        "policies": [],
    },
]


DEMO_CLAIMS = [
    # John Tan — 1 processing travel claim
    {
        "claimNo": "CLM100101",
        "policyNo": "TA300101",
        "nric": "S8834567A",
        "productName": "Travel Protect360",
        "status": "Processing",
        "claimAmount": 2850.00,
        "claimDate": "2026-02-18T00:00:00",
        "claimType": "Medical Expense",
        "description": "Emergency medical treatment for food poisoning during trip to Thailand",
    },
    # Sarah Lim — 1 approved car claim, 1 processing fraud claim
    {
        "claimNo": "CLM100201",
        "policyNo": "MP300401",
        "nric": "S9245678B",
        "productName": "Car Protect360",
        "status": "Approved",
        "claimAmount": 4200.00,
        "claimDate": "2025-12-05T00:00:00",
        "claimType": "Vehicle Repair",
        "description": "Windscreen and front bumper damage from road debris on PIE",
    },
    {
        "claimNo": "CLM100202",
        "policyNo": "CY300501",
        "nric": "S9245678B",
        "productName": "Fraud Protect360 Plus",
        "status": "Processing",
        "claimAmount": 1580.00,
        "claimDate": "2026-03-02T00:00:00",
        "claimType": "Online Fraud",
        "description": "Unauthorized online transaction on e-commerce platform",
    },
    # Raj Kumar — 1 approved maid claim
    {
        "claimNo": "CLM100301",
        "policyNo": "DY300801",
        "nric": "S7856789C",
        "productName": "Maid Protect360 PRO",
        "status": "Approved",
        "claimAmount": 3100.00,
        "claimDate": "2025-11-20T00:00:00",
        "claimType": "Medical Expense",
        "description": "Helper hospitalization for dengue fever treatment at Tan Tock Seng Hospital",
    },
    # Mei Ling Wong — 1 processing family claim
    {
        "claimNo": "CLM100401",
        "policyNo": "FA301101",
        "nric": "T0167890D",
        "productName": "Family Protect360",
        "status": "Processing",
        "claimAmount": 890.00,
        "claimDate": "2026-03-10T00:00:00",
        "claimType": "Accident Medical",
        "description": "Outpatient treatment after minor cycling accident at East Coast Park",
    },
    # David Chen — 1 approved home claim, 1 processing hospital claim
    {
        "claimNo": "CLM100501",
        "policyNo": "HC301201",
        "nric": "S7078901E",
        "productName": "Home Protect360",
        "status": "Approved",
        "claimAmount": 6800.00,
        "claimDate": "2025-10-15T00:00:00",
        "claimType": "Property Damage",
        "description": "Water damage to living room and master bedroom from burst pipe",
    },
    {
        "claimNo": "CLM100502",
        "policyNo": "HI301401",
        "nric": "S7078901E",
        "productName": "Hospital Protect360",
        "status": "Processing",
        "claimAmount": 1200.00,
        "claimDate": "2026-02-28T00:00:00",
        "claimType": "Hospital Cash",
        "description": "4-day hospitalization for minor surgery at Mount Elizabeth Hospital",
    },
]


def seed(reset: bool = False, show_only: bool = False) -> None:
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27018")
    db_name = os.getenv("DB_NAME", "bigtapp").lower()

    logger.info("Connecting to MongoDB: %s / %s", mongo_uri.split("@")[-1], db_name)
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[db_name]

    cust_col = db["demo_customers"]
    claim_col = db["demo_claims"]

    if show_only:
        logger.info("--- DEMO CUSTOMERS (%d) ---", cust_col.count_documents({}))
        for c in cust_col.find({}, {"_id": 0}):
            logger.info(
                "  %s %s | NRIC: %s | Policies: %d",
                c.get("givenName"), c.get("surname"),
                c.get("idCardNumber"),
                len(c.get("policies", [])),
            )
        logger.info("--- DEMO CLAIMS (%d) ---", claim_col.count_documents({}))
        for cl in claim_col.find({}, {"_id": 0}):
            logger.info(
                "  %s | %s | %s | $%.2f | %s",
                cl.get("claimNo"), cl.get("policyNo"),
                cl.get("status"), cl.get("claimAmount", 0),
                cl.get("claimType"),
            )
        client.close()
        return

    if reset:
        logger.warning("Dropping existing demo data...")
        cust_col.drop()
        claim_col.drop()

    existing = cust_col.count_documents({})
    if existing > 0 and not reset:
        logger.info("Demo data already exists (%d customers). Use --reset to re-seed.", existing)
        client.close()
        return

    # Insert customers
    cust_col.create_index([("idCardNumber", ASCENDING)], unique=True, name="idx_nric")
    cust_col.create_index(
        [("givenName", ASCENDING), ("surname", ASCENDING)],
        name="idx_name",
    )

    for cust in DEMO_CUSTOMERS:
        doc = {**cust, "_seeded_at": datetime.now(timezone.utc)}
        cust_col.replace_one(
            {"idCardNumber": cust["idCardNumber"]}, doc, upsert=True,
        )
    logger.info("Inserted %d demo customers", len(DEMO_CUSTOMERS))

    # Insert claims
    claim_col.create_index([("claimNo", ASCENDING)], unique=True, name="idx_claim_no")
    claim_col.create_index([("nric", ASCENDING)], name="idx_claim_nric")
    claim_col.create_index([("policyNo", ASCENDING)], name="idx_claim_policy")

    for claim in DEMO_CLAIMS:
        doc = {**claim, "_seeded_at": datetime.now(timezone.utc)}
        claim_col.replace_one(
            {"claimNo": claim["claimNo"]}, doc, upsert=True,
        )
    logger.info("Inserted %d demo claims", len(DEMO_CLAIMS))

    logger.info("=" * 55)
    logger.info("Demo data seeded successfully!")
    logger.info("  Customers : %d", cust_col.count_documents({}))
    logger.info("  Claims    : %d", claim_col.count_documents({}))
    logger.info("=" * 55)
    client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed demo customer data into MongoDB")
    parser.add_argument("--reset", action="store_true", help="Drop and re-seed all demo data")
    parser.add_argument("--show", action="store_true", help="Show current demo data without modifying")
    args = parser.parse_args()

    seed(reset=args.reset, show_only=args.show)
