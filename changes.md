## changes.md

### 2026-01-26 – Capabilities detection + loop prevention + recommendation UX

#### 1) Capabilities detection improvements (no extra LLM call)
- **File**: `nodes/supervisor.py`
  - Added a fast-path heuristic (`_is_capabilities_or_services_query`) so questions like:
    - “what are the services?”
    - “what services do you offer?”
    - “how can you help?”
    route directly to `capabilities_agent` (and clear any pending discovery/question state).
- **File**: `tools/capabilities.py`
  - Added a deterministic response for “services/capabilities” queries to avoid spending an LLM call for simple catalog answers and reduce loops.

#### 2) Intent classifier prompt broadened for “services”
- **File**: `nodes/intent.py`
  - Updated guidance so “services / what can you do” phrases are classified as `capabilities` more reliably.

#### 3) Prevent “services” from entering product-discovery loops in info flow
- **File**: `nodes/agents.py`
  - If user asks “services / what can you do”, route to `_capabilities_tool` immediately.
  - Removed the “specific vs customizable” discovery prompt from **info flow**; info now asks for product directly when needed.

#### 4) Move “specific vs customizable” question to **recommendation flow** (UX fix)
- **File**: `nodes/rec_subgraph.py`
  - When user asks for a recommendation but no product is known, the bot now asks:
    - “Would you like to find out about a specific product or something you can customize?”
  - If user indicates customizable (Choice Protect360), the subgraph shows the Choice intro and “find out more?” flow.
  - If user replies “Yes”, the bot sends the Choice base-plan + add-ons + stacked-discount explanation (matching the intended script).
  - Prevents auto-generating a tier immediately during the scripted Choice intro/follow-up (waits for the user’s next message).
  - Handles cases where the intent classifier already set product=`choice` (still triggers the Choice intro instead of jumping to a tier).

#### 5) Fix Choice Protect360 recommendation template lookup (bug fix)
- **File**: `tools/recommendation.py`
  - Normalizes product key via `_normalize_product_key()` before selecting templates/benefits.
  - Fixes the issue where `ChoiceProtect360` didn’t match YAML key `choice`, causing an empty recommendation response.

#### 6) Preserve scripted recommendation discovery messages (avoid styler rewriting)
- **File**: `nodes/styler.py`
  - Skips the styler LLM call when `product_discovery_step` or `choice_info_step` is active in recommendation, so the scripted prompts stay intact.

#### 7) Make discovery-step routing stable (avoid misroutes / stale flags)
- **File**: `nodes/intent.py`
  - Updates the intent guidance so `product_discovery_step` / `choice_info_step` are treated as continuing the **recommendation** flow (not info).
- **File**: `nodes/supervisor.py`
  - Clears `product_discovery_step` / `choice_info_step` when the user switches away from recommendation (prevents stale flags affecting later routing).
  - Restricts the “product discovery loop breaker” to `phase=info_query` only (so recommendation discovery isn’t rerouted to capabilities).

#### 8) WhatsApp mobile prefill + confirmation in policy/claim verification
- **Files**: `handlers/whatsapp.py`, `__init__.py`, `state.py`, `nodes/service_subgraph.py`
  - Passes WhatsApp sender phone into state (`channel_user_id`) without exposing it to LLMs.
  - When verification needs a mobile number, the bot offers: “I can use the number ending XXXX from WhatsApp. Is this your registered mobile?”
  - If user confirms, it uses that number; if not, it asks for a different one.

#### 9) Smarter validation-recovery handling (no loops)
- **File**: `nodes/service_subgraph.py`
  - Accepts natural replies like “try again” / “re-enter details” (not just 1/2).
  - Allows users to exit recovery and switch flows (info/recommend/compare) with simple phrases.
  - Adds clearer prompt + logging to avoid repetitive “reply with 1 or 2” loops.
  - Falls back to a lightweight LLM classifier when the user reply is ambiguous.

