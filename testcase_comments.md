## Test case comments (UX issues + API response improvements)

### Why some test cases fail (in simple terms)
In WhatsApp, users respond in **free text**, so the bot needs to:
- **Validate what it can locally** (to catch obvious mistakes and guide the user)
- **Rely on backend APIs** to confirm whether values are *actually correct* (registered / belongs to customer / acceptable for update)

If backend responses are permissive or generic, the bot may appear to “accept” invalid-looking inputs or give vague failure messages.

---

## Authentication (NRIC / First name / Last name / Mobile / Policy no)

### What the customer experiences
- If the input is clearly invalid (wrong format), the bot can prompt the user to correct it.
- If the input *looks valid* but is not the customer’s actual registered data, the bot can only know after the **validation API** responds.

### What we validate locally
- **NRIC/FIN**
- **First name**
- **Last name**
- **Mobile number**
- **Policy number**

This is **format/sanity validation**, not confirmation that the details match HL’s records.

### What would improve the bot UX (API-side)
If `validate_customer` returns **specific mismatch reasons**, the bot can respond with targeted guidance (instead of a generic “verification failed”).

Recommended API response additions:
- **error_code** (examples): `NRIC_INVALID`, `NAME_MISMATCH`, `MOBILE_MISMATCH`, `POLICY_NOT_FOUND`, `POLICY_NOT_OWNED`
- **field_errors** map (example): `{ "mobileNo": "not_registered" }`
- **user_message**: a short message safe for display to the customer

---

## Same email / same mobile update

### What the customer experiences
- If a customer submits the **same email/mobile** again, the backend may return `success=true`.
- The bot then replies “updated successfully”, which is confusing (“Nothing changed, why did it update?”).

### What would improve the bot UX (API-side)
If update APIs explicitly indicate whether the value changed, the bot can respond more accurately:
- “That email is already your registered email.”

Recommended API response additions:
- **changed**: `true | false`
- **reason** when unchanged: `no_change`
- Optionally: **current_value_masked** (e.g., last 3 chars for email, last 4 for mobile)

---

## Invalid postal code

### What the customer experiences
- If the postal code is not valid, the bot should re-ask.
- In practice, validity depends on the **postal lookup/validation endpoint** response.
  - If the API returns `success=true` for unusual values, the bot will proceed.

### What would improve the bot UX (API-side)
Recommended API response additions:
- **is_valid**: `true | false` (explicit)
- **normalized_postal**: `"123456"` (digits-only)
- If invalid: **error_code** like `POSTAL_NOT_FOUND` + a friendly `user_message`

---

## Address updates (`update_address` and `update_insured_address`)

### What the customer experiences (key UX issue)
Because WhatsApp is free text, users can accidentally type something irrelevant at an address step (e.g., emojis / greetings). If the backend accepts it, the address may be updated with bad data; if the backend rejects it but returns a generic error, the user won’t know what to fix.

### What we can do locally (today)
- **House/block number, street name, unit number**: accepted as **any non-empty string** in order to keep WhatsApp input flexible.
  - (Some fields may have additional format checks depending on the specific flow, but in general these are not “verified as real” locally.)
- **Building name**: collected explicitly and required (so the bot does not send an empty building name).

### What we still cannot confirm locally
Even if input is non-empty, the bot cannot confirm:
- the address is a real/registered address
- the combination of postal + block + street + unit + building is consistent

### What would improve the bot UX (API-side)
Recommended API response additions for address update endpoints:
- **field_errors** (example):
  - `{"unitNo": "invalid_format", "streetName": "required" }`
- **normalized_address** returned on success (so the bot can confirm what was saved)
- **error_code** for common issues (examples): `INVALID_UNIT_FORMAT`, `MISSING_BUILDING_NAME`, `ADDRESS_REJECTED`

---

## Payment update (card details)

### What the customer experiences
- The bot may accept card details that look valid (16 digits), but:
  - the card could still be invalid (checksum, expired, blocked, etc.)
  - without strong backend validation messages, failures are hard to explain to the customer

### Local validation limitations
- **No real-time card validation** (no payment gateway validation, no issuer checks).
- Card handling is based on masking + basic checks; in some “17-digit-like” inputs, the masker can still capture a **16-digit chunk**, and that may be what gets passed onward.

### What would improve the bot UX (API-side)
Recommended API response additions:
- **error_code** (examples): `CARD_INVALID`, `CARD_EXPIRED`, `CARD_TYPE_MISMATCH`
- **field_errors**: `{ "cardNo": "failed_checksum" }`
- On success: return **card_last4** and **effective_date** (so the user has confidence what was updated)

---





