import json
import httpx
from pathlib import Path

BOT_URL = "http://127.0.0.1:8080"
DATASET_DIR = Path("dataset")

def main():
    print("Starting generation of submission.jsonl...")
    
    # 1. Verify bot is healthy
    try:
        resp = httpx.get(f"{BOT_URL}/v1/healthz")
        resp.raise_for_status()
        print("Bot server is healthy and running.")
    except Exception as e:
        print(f"Error connecting to bot at {BOT_URL}: {e}")
        print("Please make sure uvicorn bot:app is running.")
        return

    # 2. Load categories
    cat_dir = DATASET_DIR / "categories"
    if cat_dir.exists():
        for f in cat_dir.glob("*.json"):
            cat = json.load(open(f))
            slug = cat.get("slug")
            print(f"Pushing category: {slug}")
            httpx.post(f"{BOT_URL}/v1/context", json={
                "scope": "category",
                "context_id": slug,
                "version": 1,
                "payload": cat,
                "delivered_at": "2026-07-03T12:00:00Z"
            })

    # 3. Load test pairs
    test_pairs_path = DATASET_DIR / "test_pairs.json"
    if not test_pairs_path.exists():
        print(f"Error: {test_pairs_path} not found. Please run generate_dataset.py first.")
        return
        
    test_pairs_data = json.load(open(test_pairs_path))
    pairs = test_pairs_data.get("pairs", [])
    print(f"Loaded {len(pairs)} test pairs.")

    submission_lines = []

    # 4. Generate composed message for each test pair
    for p in pairs:
        test_id = p["test_id"]
        trigger_id = p["trigger_id"]
        merchant_id = p["merchant_id"]
        customer_id = p["customer_id"]

        print(f"\nProcessing {test_id}: Trigger={trigger_id}, Merchant={merchant_id}")

        # Load and push merchant context
        m_path = DATASET_DIR / "merchants" / f"{merchant_id}.json"
        if not m_path.exists():
            print(f"Merchant file {m_path} not found, skipping.")
            continue
        merchant = json.load(open(m_path))
        httpx.post(f"{BOT_URL}/v1/context", json={
            "scope": "merchant",
            "context_id": merchant_id,
            "version": 1,
            "payload": merchant,
            "delivered_at": "2026-07-03T12:00:00Z"
        })

        # Load and push customer context if present
        if customer_id:
            c_path = DATASET_DIR / "customers" / f"{customer_id}.json"
            if c_path.exists():
                customer = json.load(open(c_path))
                httpx.post(f"{BOT_URL}/v1/context", json={
                    "scope": "customer",
                    "context_id": customer_id,
                    "version": 1,
                    "payload": customer,
                    "delivered_at": "2026-07-03T12:00:00Z"
                })

        # Load and push trigger context
        t_path = DATASET_DIR / "triggers" / f"{trigger_id}.json"
        if not t_path.exists():
            print(f"Trigger file {t_path} not found, skipping.")
            continue
        trigger = json.load(open(t_path))
        httpx.post(f"{BOT_URL}/v1/context", json={
            "scope": "trigger",
            "context_id": trigger_id,
            "version": 1,
            "payload": trigger,
            "delivered_at": "2026-07-03T12:00:00Z"
        })

        # Trigger tick
        tick_resp = httpx.post(f"{BOT_URL}/v1/tick", json={
            "now": "2026-07-03T12:05:00Z",
            "available_triggers": [trigger_id]
        }, timeout=60.0)
        
        tick_data = tick_resp.json()
        actions = tick_data.get("actions", [])
        if actions:
            action = actions[0]
            submission_line = {
                "test_id": test_id,
                "body": action["body"],
                "cta": action["cta"],
                "send_as": action["send_as"],
                "suppression_key": action["suppression_key"],
                "rationale": action["rationale"]
            }
            submission_lines.append(submission_line)
            print(f"Result composed successfully: \"{action['body'][:60]}...\"")
        else:
            print(f"Warning: No action generated for {test_id}.")

    # 5. Write submission.jsonl
    with open("submission.jsonl", "w", encoding="utf-8") as f:
        for line in submission_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nDone! Wrote {len(submission_lines)} lines to submission.jsonl.")

if __name__ == "__main__":
    main()
