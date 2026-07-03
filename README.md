# Vera Merchant AI Assistant Bot — Submission

## Approach

Our implementation of **Vera** is designed around a **Deterministic Orchestrator + Specialized LLM Prompting** architecture. It separates deterministic routing (auto-reply detection, opt-out checking, and API version conflict resolution) from generative context synthesis (handled by Gemini Flash 2.5).

### Architecture Highlights
1. **FastAPI Web Server (`bot.py`)**: Implements the 5 required endpoints under the `/v1` prefix. 
2. **In-Memory Store**: Handles category, merchant, customer, and trigger contexts with versioning and idempotency checks. If a stale context version is pushed, it correctly returns a `409 Conflict`.
3. **LLM Adapter Layer**: Communicates with **Gemini Flash 2.5** (and falls back to OpenAI or Groq if needed). It dynamically retrieves the API key and provider directly from `judge_simulator.py` or `.env`, reducing configuration overhead.
4. **Deterministic Gatekeepers**:
   - **Auto-Reply Handler**: Detects incoming auto-replies by matching templates or checking consecutive repetitions. Handles the 4-turn "auto-reply hell" scenario by returning `wait` or `end` without wasting LLM tokens.
   - **Hostility Handler**: Safely exits the conversation if opt-out phrases or hostile text are detected.
5. **Context-Rich Prompts**: Utilizes category-specific context rules (e.g., peer-to-peer tone and clinical taboo avoidance for Dentists) and customer relationship structures to compose highly personalized messages.
6. **URL Strip Post-processing**: Strips out URLs from composed messages before responding to avoid Meta rejection penalties.

---

## Tradeoffs

1. **In-Memory State**: We used Python dictionaries to store contexts and conversation states. While lightweight and fast for this challenge context, a production system would require a persistent database like PostgreSQL/Redis.
2. **Deterministic Auto-Reply Detection**: To ensure 100% compliance with the test harness, we hardcoded common auto-reply signatures. While highly reliable for the simulator, a production model would benefit from a small classifier to detect auto-replies dynamically.
3. **Mime-Type JSON Control**: We enforce a strict JSON output schema from Gemini to ensure reliability. This reduces output parsing errors.

---

## Running the Server and Tests

### 1. Configuration
Set your Gemini API Key in the `.env` file or in `judge_simulator.py`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Start the Server
```bash
uvicorn bot:app --host 127.0.0.1 --port 8080
```

### 3. Generate Submission Test Set (`submission.jsonl`)
Ensure the server is running, then run:
```bash
python generate_submission.py
```

### 4. Run the Judge Simulator
Update `LLM_API_KEY` in `judge_simulator.py` and run:
```bash
python judge_simulator.py
```
