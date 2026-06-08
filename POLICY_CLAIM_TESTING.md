# Policy and Claim Testing Guide (WhatsApp)

This document provides testing scenarios for the Policy and Claim functionalities of the bot via WhatsApp.

> **⚠️ IMPORTANT**: The client has provided only **ONE** single mock record. 
> **Please keep track of the information that you change during testing.** 
> If you update the address or phone number, subsequent tests must use the *updated* information until you reset it.

## Prerequisites

1.  **WhatsApp Access**: Ensure you have access to the testing WhatsApp bot number.
2.  **Test Data**: Use the following details for authentication and verification.

| Field | Value | Notes |
| :--- | :--- | :--- |
| **NRIC** | `S8978229E` | Fixed identifier |
| **Name** | `WL TIO` | Surname: `TIO`, Given Name: `WL` |
| **Phone Number** | `81384997` | **Use this exact number**. The bot tracks the phone number for customer validation. |
| **Policy Number** | `DY300318` | Use this policy number for all inquiries. |

---

## Test Scenarios

### 1. Authentication & Identity Verification
Before accessing policy details, the bot must verify the user's identity by collecting specific credentials.

**Flow:**
1.  **User**: Send `Hi` or `Hello` to start the session.
2.  **Bot**: Responds with a greeting.
3.  **User**: `I want to check my policy details` (or similar intent).
4.  **Bot**: The bot will ask for the following details one by one (if not already provided):
    *   **NRIC/FIN**: Enter `S8978229E`.
    *   **First Name**: Enter `WL`.
    *   **Last Name**: Enter `TIO`.
    *   **Mobile Number**: Enter `81384997` (Must match registered number).
    *   **Policy Number**: Enter `DY300318`.
5.  **Bot**: Validates the information against the backend.
6.  **Bot**: Confirms identity ("Thanks WL! I've verified your identity.") and proceeds to the request.

---

### 2. Policy Status Check
Check the status and details of your policies.

**Flow:**
1.  **User**: `Check my policy status`.
2.  **Bot**: (If authenticated) Retrieves your policies.
3.  **Bot**: Displays a list of policies (Active/Lapsed) or details of a specific policy if mentioned.
    *   *Example Output*: 
        *   ✅ **DY300318** - Home Protect
        *   Status: Active | Ends: 01 Jan 2025
4.  **User**: `Show me details for DY300318`.
5.  **Bot**: distinct details including start/end dates and product name.

---

### 3. Claim Status
Check the status of your claims.

**Flow:**
1.  **User**: `Check my claim status` or `I have a claim question`.
2.  **Bot**: Retrieves claims associated with your NRIC.
3.  **Bot**: Displays a list of claims with their status.
    *   *Example Output*:
        *   ⏳ Policy **DY300318** - Status: Processing

---

### 4. Update Email Address
Change your registered email address.

**Flow:**
1.  **User**: `I want to update my email address`.
2.  **Bot**: `What would you like your new email address to be?`
3.  **User**: Enter a new email, e.g., `tester@example.com`.
4.  **Bot**: Calls API to update email.
5.  **Bot**: Returns success message ("✅ Your email has been updated successfully!").

---

### 5. Update Mobile Number
Change your registered mobile number.

**Flow:**
1.  **User**: `I want to change my phone number`.
2.  **Bot**: `What would you like your new mobile number to be?`
3.  **User**: Enter a new valid Singapore mobile number (e.g., `91234567`).
4.  **Bot**: Calls API to update mobile number.
5.  **Bot**: Returns success message.
    *   *Note: Ensure you track this change as the bot tracks the user by phone number.*

---

### 6. Home Protect Insured Address Update
Update the insured address for the Home Protect policy.

**Flow:**
1.  **User**: `I want to update my home address`.
2.  **Bot**: Asks which policy (if multiple) or confirms the policy `DY300318`.
3.  **Bot**: `What is the new postal code for the insured property?`
4.  **User**: `089057` (Valid) or `730600`.
5.  **Bot**: `What is the block or house number?`
6.  **User**: e.g., `600`.
7.  **Bot**: `What is the street name?`
8.  **User**: e.g., `ADAM ROAD`.
9.  **Bot**: `What is the unit number?`
10. **User**: e.g., `#01-01`.
11. **Bot**: Confirms details and processes update.
12. **Bot**: Returns success message.

---

### 7. Payment Info Update
Update credit card details for premium payments.

**Flow:**
1.  **User**: `I want to update my payment details`.
2.  **Bot**: Asks for the policy number (if not already in context).
3.  **User**: `DY300318`.
4.  **Bot**: `Please enter your new credit/debit card number.`
5.  **User**: `4111111111111111` (Test Visa).
6.  **Bot**: `What is the card expiry date?`
7.  **User**: `01/10/2029`.
8.  **Bot**: `What type of card is this? (VISA, Mastercard, or AMEX)`
9.  **User**: `VISA`.
10. **Bot**: Confirms the update with masked card details.

---

### 8. General Address Change (Mailing Address)
Update the customer's mailing address (distinct from insured address).

**Flow:**
1.  **User**: `I want to change my mailing address`.
2.  **Bot**: `What is your new postal code?`
3.  **User**: `089057`.
4.  **Bot**: (Validates postal code) `What is your block or house number?`
5.  **User**: `Blk 123`.
6.  **Bot**: `What is your street name?`
7.  **User**: `Test Street`.
8.  **Bot**: `What is your unit number?`
9.  **User**: `#10-10`.
10. **Bot**: Confirms update.

---

### ⚠️ Troubleshooting & Notes
- **Phone Number Tracking**: The bot relies on the phone number `81384997` for validation. Ensure you are either testing from this number (if whitelisted) or supplying it correctly when prompted.
- **Postal Codes**: Always use valid Singapore postal codes (6 digits). 
    - Valid examples: `089057`, `730600`.
- **Session Reset**: If you get stuck, type `Hi` to reset the session.
