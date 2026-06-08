# Policy And Claims Demo Data

This bot already has a mock customer and claims database for policy/claims testing.

Source of truth:
- `scripts/seed_demo_data.py`
- MongoDB collections: `demo_customers`, `demo_claims`

## Seeded Demo Customers

### 1. John Tan
- NRIC: `S8834567A`
- First Name: `John`
- Last Name: `Tan`
- Email: `john.tan@gmail.com`
- Mobile: `91234567`
- Policies:
  - `TA300101` - Travel Protect360 - Active
  - `HC300201` - Home Protect360 - Active
  - `FA300301` - Family Protect360 - Lapsed

### 2. Sarah Lim
- NRIC: `S9245678B`
- First Name: `Sarah`
- Last Name: `Lim`
- Email: `sarah.lim@outlook.com`
- Mobile: `82345678`
- Policies:
  - `MP300401` - Car Protect360 - Active
  - `CY300501` - Fraud Protect360 Plus - Active
  - `HI300601` - Hospital Protect360 - Active

### 3. Raj Kumar
- NRIC: `S7856789C`
- First Name: `Raj`
- Last Name: `Kumar`
- Email: `raj.kumar@yahoo.com`
- Mobile: `93456789`
- Policies:
  - `ES300701` - Early Protect360 Plus - Active
  - `DY300801` - Maid Protect360 PRO - Active
  - `CK300901` - ChoiceProtect360 - Lapsed

### 4. Mei Ling Wong
- NRIC: `T0167890D`
- First Name: `Mei Ling`
- Last Name: `Wong`
- Email: `meiling.wong@gmail.com`
- Mobile: `84567890`
- Policies:
  - `TB301001` - Travel Protect360 - Active
  - `FA301101` - Family Protect360 - Active

### 5. David Chen
- NRIC: `S7078901E`
- First Name: `David`
- Last Name: `Chen`
- Email: `david.chen@hotmail.com`
- Mobile: `95678901`
- Policies:
  - `HC301201` - Home Protect360 - Active
  - `MP301301` - Car Protect360 - Active
  - `HI301401` - Hospital Protect360 - Active

## Seeded Demo Claims

### John Tan
- `CLM100101` - `TA300101` - Travel Protect360 - Processing

### Sarah Lim
- `CLM100201` - `MP300401` - Car Protect360 - Approved
- `CLM100202` - `CY300501` - Fraud Protect360 Plus - Processing

### Raj Kumar
- `CLM100301` - `DY300801` - Maid Protect360 PRO - Approved

### Mei Ling Wong
- `CLM100401` - `FA301101` - Family Protect360 - Processing

### David Chen
- `CLM100501` - `HC301201` - Home Protect360 - Approved
- `CLM100502` - `HI301401` - Hospital Protect360 - Processing

## Suggested Policy Test Prompts

1. Check my policy status.
2. Show me my policy details.
3. Show me details for policy TA300101.
4. Show me details for policy CY300501.
5. Show me details for policy DY300801.
6. Is my policy HC301201 active?
7. Which policies do I have under my NRIC?
8. I want to check my policy details for MP300401.
9. I want to update my email address.
10. I want to change my mobile number.
11. I want to update my mailing address.
12. I want to update my home address.
13. I want to update my insured property address.
14. I want to change the payment details for policy HC301201.
15. I want to update the policy linked to Sarah Lim.

## Validation Note

For the current policy/claims service flow, the bot may ask for these verification fields before proceeding:
- First name
- Last name
- Registered email address
- Registered mobile number

Use the seeded values above when testing end to end.

## Suggested Claims Test Prompts

1. Check my claim status.
2. Show me my latest claim.
3. What is the status of claim CLM100101?
4. What is the status of claim CLM100202?
5. What is the status of claim CLM100502?
6. Do I have any claims under policy TA300101?
7. Do I have any claims under policy CY300501?
8. Show me claims linked to policy HC301201.
9. I want to check the claim for my Fraud Protect360 Plus policy.
10. I want to know whether my Home Protect360 claim is approved.
11. I have a question about my claim for policy MP300401.
12. Can you check whether my hospital cash claim is still processing?
13. What is the status of my maid insurance claim?
14. Show me all approved claims under my profile.
15. Show me all processing claims under my profile.
