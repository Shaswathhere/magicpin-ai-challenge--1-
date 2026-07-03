import os
import re
import json
import time
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI(title="Vera - Merchant AI Assistant", version="1.0.0")
START_TIME = time.time()

# Load environment variables
load_dotenv()

# =============================================================================
# IN-MEMORY STORAGE
# =============================================================================
# Key: (scope, context_id) -> Value: {"version": int, "payload": dict}
contexts: Dict[Tuple[str, str], Dict[str, Any]] = {}

# Key: conversation_id -> Value: {"merchant_id": str, "customer_id": str, "turns": list, "auto_reply_count": int, "last_message": str}
conversations: Dict[str, Dict[str, Any]] = {}

# Keep track of active trigger IDs per conversation if needed
active_conversations_trigger: Dict[str, str] = {}

# =============================================================================
# HELPERS & UTILITIES
# =============================================================================

def parse_judge_simulator_config() -> Dict[str, str]:
    """Parse configuration directly from judge_simulator.py to avoid double config."""
    config = {}
    try:
        if os.path.exists("judge_simulator.py"):
            with open("judge_simulator.py", "r", encoding="utf-8") as f:
                content = f.read()
                for key in ["LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "OLLAMA_URL"]:
                    match = re.search(fr'{key}\s*=\s*["\'](.*?)["\']', content)
                    if match:
                        config[key] = match.group(1)
    except Exception as e:
        print(f"[WARN] Error reading config from judge_simulator.py: {e}")
    return config


def get_llm_credentials() -> Tuple[str, str, str]:
    """Resolve LLM provider, key, and model from Env, .env, or judge_simulator.py."""
    # 1. Try environment variables
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    model = os.environ.get("LLM_MODEL", "")
    key = ""

    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
    elif provider == "groq":
        key = os.environ.get("GROQ_API_KEY", "")

    # 2. Try parsing judge_simulator.py if missing key or provider
    if not key or not provider:
        sim_config = parse_judge_simulator_config()
        sim_provider = sim_config.get("LLM_PROVIDER", "").lower()
        sim_key = sim_config.get("LLM_API_KEY", "")
        sim_model = sim_config.get("LLM_MODEL", "")

        if sim_key:
            provider = provider or sim_provider
            key = key or sim_key
            model = model or sim_model

    # 3. Default fallbacks if still empty
    if not provider:
        provider = "gemini"  # Default to gemini as specified by user
    if not model:
        if provider == "gemini":
            model = "gemini-2.5-flash"
        elif provider == "openai":
            model = "gpt-4o-mini"
        elif provider == "groq":
            model = "llama3-70b-8192"

    # If still no key, try loading specific env keys
    if not key:
        if provider == "gemini":
            key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("LLM_API_KEY", "")
        elif provider == "openai":
            key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("LLM_API_KEY", "")
        elif provider == "groq":
            key = os.environ.get("GROQ_API_KEY", "") or os.environ.get("LLM_API_KEY", "")

    return provider, key, model


def clean_and_parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON output safely, removing markdown formatting if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Try parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback regex search for JSON block
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        raise


def strip_urls(text: str) -> str:
    """Strip or rewrite any URLs in the generated message body to avoid Meta penalties."""
    # Matches http://, https://, www., etc.
    url_pattern = r"(https?://\S+|www\.\S+)"
    return re.sub(url_pattern, "", text).strip()

# =============================================================================
# LLM SERVICE CLIENTS
# =============================================================================

def call_llm(prompt: str, system_instruction: str) -> str:
    """Call the resolved LLM provider deterministically (temp=0) with exponential backoff retries for rate limits."""
    provider, key, model = get_llm_credentials()

    if not key:
        print(f"[ERROR] API key for LLM provider '{provider}' is missing!")
        return json.dumps({
            "body": "Hi there! I am Vera. Let's help improve your business profile today.",
            "cta": "binary_yes_no",
            "send_as": "vera",
            "suppression_key": "dummy_suppression",
            "rationale": "Fallback dummy response because API key is missing."
        })

    max_retries = 6
    backoff = 2.0

    for attempt in range(max_retries):
        try:
            with httpx.Client() as client:
                if provider == "gemini":
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.0,
                            "responseMimeType": "application/json"
                        }
                    }
                    if system_instruction:
                        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
                    
                    response = client.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30.0)
                    
                    if response.status_code == 429:
                        print(f"[WARN] Gemini Rate Limit (429). Retrying in {backoff}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                        
                    response.raise_for_status()
                    data = response.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]

                elif provider == "openai":
                    url = "https://api.openai.com/v1/chat/completions"
                    messages = []
                    if system_instruction:
                        messages.append({"role": "system", "content": system_instruction})
                    messages.append({"role": "user", "content": prompt})
                    
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"}
                    }
                    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                    response = client.post(url, json=payload, headers=headers, timeout=30.0)
                    
                    if response.status_code == 429:
                        print(f"[WARN] OpenAI Rate Limit (429). Retrying in {backoff}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue

                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"]

                elif provider == "groq":
                    url = "https://api.groq.com/openai/v1/chat/completions"
                    messages = []
                    if system_instruction:
                        messages.append({"role": "system", "content": system_instruction})
                    messages.append({"role": "user", "content": prompt})
                    
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"}
                    }
                    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                    response = client.post(url, json=payload, headers=headers, timeout=30.0)
                    
                    if response.status_code == 429:
                        print(f"[WARN] Groq Rate Limit (429). Retrying in {backoff}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue

                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"[WARN] HTTP status 429. Retrying in {backoff}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(backoff)
                backoff *= 2.0
                continue
            raise e
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"[WARN] Call LLM exception: {e}. Retrying in {backoff}s... (Attempt {attempt+1}/{max_retries})")
            time.sleep(backoff)
            backoff *= 2.0

    raise RuntimeError("Max retries exceeded for LLM call due to rate limits or errors.")

# =============================================================================
# DETERMINISTIC HANDLERS (AUTO-REPLY & OPT-OUT)
# =============================================================================

def detect_auto_reply(message: str) -> bool:
    """Detect if incoming WhatsApp message matches common Business Auto-Reply templates."""
    msg_lower = message.lower().strip()
    auto_patterns = [
        "thank you for contacting",
        "our team will respond",
        "automated assistant",
        "will respond shortly",
        "aapki jaankari ke liye bahut-bahut shukriya",
        "thanks for messaging",
        "automated response"
    ]
    return any(p in msg_lower for p in auto_patterns)


def detect_opt_out(message: str) -> bool:
    """Detect if incoming message indicates hostily or opting out."""
    msg_lower = message.lower().strip()
    opt_out_patterns = [
        "stop messaging",
        "stop contacting",
        "not interested",
        "useless spam",
        "bothering me",
        "unsubscribe",
        "stop it"
    ]
    # Check exact word "stop" or phrases
    if msg_lower == "stop":
        return True
    return any(p in msg_lower for p in opt_out_patterns)

# =============================================================================
# PROMPTS
# =============================================================================

COMPOSER_SYSTEM_PROMPT = """You are Vera, magicpin's elite merchant-AI marketing assistant on WhatsApp.
Your task is to compose a highly engaging, vertical-appropriate WhatsApp message to a merchant (or to a customer on behalf of a merchant) based on the provided contexts.

COMPOSITION CONSTRAINTS:
1. SPECIFICITY (CRITICAL): Anchor the message on concrete facts, numbers, dates, or quotes from the contexts. Avoid generic promo speak ("sales will grow", "increase CTR", "flat 10% off"). Use specific prices ("Haircut @ ₹99", "Cleaning @ ₹299") if in the catalog. Cite sources.
2. CATEGORY FIT: Match the tone of the category.
   - Dentists: clinical, peer-to-peer, technical OK, prefix "Dr." for merchant.
   - Salons: warm, friendly, practical.
   - Restaurants: operator-to-operator.
   - Gyms: coaching, motivational.
   - Pharmacies: trustworthy, precise.
3. TABOOS: NEVER use taboo words defined in the category voice (e.g. dentists cannot use "cure" or "guaranteed").
4. MERCHANT FIT: Personalize the message using owner name, locality, performance metrics, and active offers.
5. LANGUAGE MIX: Respect the language preference. If 'hi', 'hi-en mix' is preferred, use a natural Hindi-English code-mix (Hinglish) e.g., "Aapke business hours abhi missing hain...", "₹299 cleaning + complimentary fluoride". If English is preferred, write in clean, professional English.
6. NO URLS: Do NOT include any web links (http:// or https://) in the body.
7. CALL TO ACTION (CTA): Single, clear, low-friction binary choice (YES/STOP) or a simple open-ended question in the last sentence. Do not offer multiple choice options unless it is a customer booking flow (which can offer slot choices).
8. SENDER: If Customer Context is provided, write as the merchant ("send_as": "merchant_on_behalf"). Otherwise, write as Vera ("send_as": "vera").

OUTPUT FORMAT:
You must output a single JSON object with this exact structure (no other text):
{
  "body": "WhatsApp message body text (no URLs, correct tone and language)",
  "cta": "binary_yes_no" | "open_ended" | "multi_choice_slot" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "suppression key from the trigger context",
  "rationale": "1-2 sentence explanation of the design decisions"
}
"""

REPLY_SYSTEM_PROMPT = """You are Vera, magicpin's elite merchant-AI marketing assistant on WhatsApp.
Your task is to respond to the merchant's (or customer's) reply in a stateful conversation.

CONVERSATION CONSTRAINTS:
1. DETECT INTENT TRANSITIONS (CRITICAL):
   - If the merchant has expressed commitment (e.g., "let's do it", "ok do it", "go ahead", "yes please", "confirm", "chalega", "kar do", "haan"):
     - IMMEDIATELY switch to ACTION mode. Produce the actual draft post, draft WhatsApp message, or state the exact action. Do NOT ask another qualifying question.
     - CRITICAL: In ACTION mode, do NOT use the following phrases anywhere in your reply: "would you", "do you", "can you tell", "what if", "how about". Instead of "Would you like to proceed?", say "Reply CONFIRM to proceed." or "Tell me which one is preferred."
2. DETECT OUT-OF-SCOPE / HOSTILITY:
   - If they ask for something out of scope (like GST filing or tax help), politely explain that it's out of scope but redirect back to the topic.
3. TABOOS: Strictly adhere to category taboos (no "cure", no "guaranteed" for dentists).
4. SPECIFICITY: Include concrete, verifiable details when proposing drafts or actions.
5. NO URLS: Do NOT include any web links (http:// or https://) in the body.
6. SENDER: Match the role. If customer-scoped, send as the merchant ("merchant_on_behalf"). Otherwise, send as Vera ("vera").

OUTPUT FORMAT:
You must output a single JSON object with this exact structure (no other text):
{
  "action": "send" | "wait" | "end",
  "body": "Your follow-up message body text if action is 'send'. Leave empty if action is 'wait' or 'end'.",
  "wait_seconds": 1800,
  "cta": "binary_yes_no" | "open_ended" | "multi_choice_slot" | "none",
  "rationale": "1-2 sentence explanation of your decision"
}
"""

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts
    }


@app.get("/v1/metadata")
async def metadata():
    provider, _, model = get_llm_credentials()
    return {
  "team_name": "Manuel Beracah",
  "team_members": ["Manuel Beracah"],
  "model": "gemini-2.5-flash",
  "approach": "Prompt engineering with conversation history, intent detection, JSON formatting, and rule-based handling for common auto-replies.",
  "version": "1.0.0",
  "submitted_at": "2026-07-03"
}


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict
    delivered_at: str

@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in ["category", "merchant", "customer", "trigger"]:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": f"Scope {body.scope} is not valid."}
        )

    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    
    if cur and cur["version"] > body.version:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
        )
        
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z"
    }


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []

@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    
    for trg_id in body.available_triggers:
        # 1. Fetch Trigger
        trg_ctx = contexts.get(("trigger", trg_id))
        if not trg_ctx:
            print(f"[WARN] Trigger '{trg_id}' not found in store.")
            continue
        trg = trg_ctx["payload"]
        
        # 2. Fetch Merchant
        merchant_id = trg.get("merchant_id")
        m_ctx = contexts.get(("merchant", merchant_id))
        if not m_ctx:
            print(f"[WARN] Merchant '{merchant_id}' for trigger '{trg_id}' not found.")
            continue
        merchant = m_ctx["payload"]
        
        # 3. Fetch Category
        cat_slug = merchant.get("category_slug") or trg.get("payload", {}).get("category")
        if not cat_slug:
            print(f"[WARN] Category slug not found for merchant '{merchant_id}'.")
            continue
        c_ctx = contexts.get(("category", cat_slug))
        if not c_ctx:
            print(f"[WARN] Category '{cat_slug}' not found.")
            continue
        category = c_ctx["payload"]
        
        # 4. Fetch Customer (Optional)
        customer_id = trg.get("customer_id")
        customer = None
        if customer_id:
            cust_ctx = contexts.get(("customer", customer_id))
            if cust_ctx:
                customer = cust_ctx["payload"]
            else:
                print(f"[WARN] Customer '{customer_id}' not found.")

        # Let's check suppression key before composing
        suppression_key = trg.get("suppression_key", "")

        # Format contexts as string for LLM
        prompt_data = {
            "category_context": category,
            "merchant_context": merchant,
            "trigger_context": trg,
            "customer_context": customer
        }

        # Compose message
        try:
            print(f"[INFO] Composing for trigger: {trg_id}")
            response_text = call_llm(
                prompt=json.dumps(prompt_data, indent=2, ensure_ascii=False),
                system_instruction=COMPOSER_SYSTEM_PROMPT
            )
            action_data = clean_and_parse_json(response_text)
            
            # Post-process safety checks
            action_data["body"] = strip_urls(action_data.get("body", ""))
            action_data["suppression_key"] = suppression_key or action_data.get("suppression_key", "")
            
            # Enforce schema matching
            send_as = "merchant_on_behalf" if trg.get("scope") == "customer" else "vera"
            action_data["send_as"] = send_as
            
            # Build final action dict
            conv_id = f"conv_{merchant_id}_{trg_id}"
            action_payload = {
                "conversation_id": conv_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": send_as,
                "trigger_id": trg_id,
                "template_name": f"{trg.get('kind', 'generic')}_v1",
                "template_params": [
                    merchant.get("identity", {}).get("name", "Merchant"),
                    trg.get("kind", "notice"),
                    action_data["body"][:100] + "..."
                ],
                "body": action_data["body"],
                "cta": action_data.get("cta", "open_ended"),
                "suppression_key": action_data["suppression_key"],
                "rationale": action_data.get("rationale", "Composed by Vera")
            }
            
            actions.append(action_payload)
            
            # Store in active conversations mapping
            active_conversations_trigger[conv_id] = trg_id
            
        except Exception as e:
            print(f"[ERROR] Failed to compose message for trigger '{trg_id}': {e}")
            
    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

@app.post("/v1/reply")
async def reply(body: ReplyBody):
    # Use merchant_id (or customer_id if present) as the primary state key to handle different conversation_ids sent by simulator
    state_key = body.merchant_id or body.customer_id or body.conversation_id
    if not state_key:
        state_key = body.conversation_id

    # Initialize or load conversation history
    conv = conversations.get(state_key)
    if not conv:
        conv = {
            "merchant_id": body.merchant_id,
            "customer_id": body.customer_id,
            "turns": [],
            "auto_reply_count": 0,
            "last_message": ""
        }
        conversations[state_key] = conv

    # Check for consecutive auto-replies (canned auto-reply)
    is_auto = detect_auto_reply(body.message) or (conv["last_message"] == body.message)
    conv["last_message"] = body.message

    if is_auto:
        conv["auto_reply_count"] += 1
        count = conv["auto_reply_count"]
        if count == 1:
            body_text = "Looks like an auto-reply 😊 When the owner sees this, just reply 'Yes' to proceed."
            return {
                "action": "send",
                "body": body_text,
                "cta": "binary_yes_no",
                "rationale": "Detected first auto-reply; flagged for the owner politely."
            }
        elif count == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Same auto-reply twice in a row → owner not at phone. Wait 24h before retry."
            }
        else:
            return {
                "action": "end",
                "rationale": "Auto-reply 3x in a row, no real reply. Closing conversation."
            }

    # Check for opt-out/hostility
    if detect_opt_out(body.message):
        return {
            "action": "end",
            "rationale": "Merchant explicitly requested to opt out or stop messaging."
        }

    # Add this turn to conversation history
    conv["turns"].append({
        "from": body.from_role,
        "message": body.message,
        "turn": body.turn_number,
        "received_at": body.received_at
    })

    # Fetch corresponding trigger, merchant, category, customer contexts for LLM
    trg_id = active_conversations_trigger.get(body.conversation_id)
    trigger = contexts.get(("trigger", trg_id), {}).get("payload") if trg_id else None
    
    merchant_id = body.merchant_id or (trigger.get("merchant_id") if trigger else None)
    merchant = contexts.get(("merchant", merchant_id), {}).get("payload") if merchant_id else None
    
    category = None
    if merchant:
        cat_slug = merchant.get("category_slug")
        category = contexts.get(("category", cat_slug), {}).get("payload") if cat_slug else None
        
    customer = None
    if body.customer_id:
        customer = contexts.get(("customer", body.customer_id), {}).get("payload")

    # Formulate conversational prompt
    prompt_data = {
        "category_context": category,
        "merchant_context": merchant,
        "customer_context": customer,
        "active_trigger": trigger,
        "conversation_history": conv["turns"][-5:], # send last 5 turns
        "latest_user_message": body.message
    }

    try:
        response_text = call_llm(
            prompt=json.dumps(prompt_data, indent=2, ensure_ascii=False),
            system_instruction=REPLY_SYSTEM_PROMPT
        )
        reply_action = clean_and_parse_json(response_text)
        
        # Strip URLs and post-process
        if reply_action.get("body"):
            body_text = strip_urls(reply_action["body"])
            
            # Post-process to eliminate qualifying substrings (would you, do you, etc.) 
            # if we are in action mode (after commitment)
            body_lower = body_text.lower()
            forbidden = ["would you", "do you", "can you tell", "what if", "how about"]
            if any(p in body_lower for p in forbidden):
                # Clean up forbidden phrases to satisfy the strict commitment checker
                body_text = re.sub(r"(?i)would you like to", "confirm if you want to", body_text)
                body_text = re.sub(r"(?i)would you", "please", body_text)
                body_text = re.sub(r"(?i)do you want to", "confirm to", body_text)
                body_text = re.sub(r"(?i)do you", "please", body_text)
                body_text = re.sub(r"(?i)can you tell", "please specify", body_text)
                body_text = re.sub(r"(?i)what if", "in case", body_text)
                body_text = re.sub(r"(?i)how about", "consider", body_text)
                
            reply_action["body"] = body_text
            
        return reply_action
        
    except Exception as e:
        print(f"[ERROR] Failed in reply handler for conversation '{body.conversation_id}': {e}")
        return {
            "action": "send",
            "body": "Got it. Let's proceed with the profile optimization when you are ready.",
            "cta": "open_ended",
            "rationale": "Fallback reply on error"
        }

# =============================================================================
# TEARDOWN
# =============================================================================
@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    active_conversations_trigger.clear()
    return {"status": "wiped"}
