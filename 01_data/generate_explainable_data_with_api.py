#!/usr/bin/env python3
"""从训练集抽样并用 LLM API 生成解释字段（需 OPENAI_API_KEY）。仅供学术研究，禁止用于非法用途。"""

import json
import random
import sys
import concurrent.futures
from concurrent.futures import TimeoutError
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI  # pip install openai

import os

REPO_ROOT = Path(__file__).resolve().parents[1]
API_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = os.environ.get("EXPLAIN_API_MODEL", "gpt-4o-mini")

INPUT_DATASET = REPO_ROOT / "data" / "dataset_scamnet_5000.json"
OUTPUT_DATASET = REPO_ROOT / "data" / "dataset_explainable_200.json"
SAMPLE_SIZE = 200
MAX_WORKERS = 5

SYSTEM_PROMPT = """You are a cybersecurity expert. Your task is to explain WHY a website is classified as 'Legitimate' or 'Scam'.
You will receive the website's raw features (URL, Content, External Links) and its correct classification Label.
You must generate a concise, professional reasoning analysis in the specific format required below.

**Format Requirement:**
You must output TWO sections strictly following this Markdown format:

## Website's Content and External Links:
[Analyze the body text, presence/absence of external links, contact info, or suspicious language here.]

## Miscellaneous:
[Analyze the URL structure, domain name patterns, or any other anomalies here.]

**Rules:**
1. Do NOT output the label again. Only output the reasoning sections.
2. Be specific to the provided content. Do not hallucinate WHOIS data if it is not provided.
3. If the label is 'Scam', focus on red flags (lack of links, urgent language, weird URL).
4. If the label is 'Legit', focus on trust signals (navigational links, clear branding, consistent URL).
"""

def generate_reasoning(client, sample, index):
    """
    Calls the API to generate reasoning for a single sample.
    """
    original_input = sample['input']
    original_output = sample['output'] # e.g. "Label: scam"
    
    # Extract the label to guide the LLM
    label_hint = "SCAM" if "Label: scam" in original_output else "LEGITIMATE"
    
    user_message = f"""
    **Website Data:**
    {original_input}

    **Correct Classification:** {label_hint}

    Please provide the reasoning for this classification based on the data above.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=512,
            timeout=90,  # 防卡死：单请求超时
        )
        reasoning = response.choices[0].message.content.strip()
        
        # Combine Original Label + New Reasoning
        # This aligns with ScamNet Step 2 format
        new_output = f"{original_output}\n{reasoning}"
        
        return {
            "input": original_input,
            "output": new_output
        }
    except Exception as e:
        print(f"\n[Error] Sample {index} failed: {e}")
        return None

def main():
    print("=" * 60)
    print("ScamNet Step 2: Generating Synthetic Reasoning Data via API")
    print("=" * 60)

    # 1. Load Data
    if not INPUT_DATASET.exists():
        print(f"[Error] Input file not found: {INPUT_DATASET}")
        sys.exit(1)

    print(f"Loading input dataset: {INPUT_DATASET}")
    with open(INPUT_DATASET, 'r') as f:
        full_data = json.load(f)

    # 2. Random Sample
    print(f"Randomly sampling {SAMPLE_SIZE} items from {len(full_data)} total samples...")
    if len(full_data) > SAMPLE_SIZE:
        selected_samples = random.sample(full_data, SAMPLE_SIZE)
    else:
        selected_samples = full_data

    if not API_KEY:
        print("[Error] 未设置 OPENAI_API_KEY。请 export OPENAI_API_KEY=... 后重试。")
        sys.exit(1)

    # 3. Initialize API Client
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"[Error] Failed to initialize OpenAI client: {e}")
        sys.exit(1)

    # 4. Process with API (Threaded)
    print(f"Starting API generation (Workers: {MAX_WORKERS})...")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_reasoning, client, sample, i): i for i, sample in enumerate(selected_samples)}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(selected_samples), unit="sample"):
            try:
                result = future.result(timeout=120)  # 防止某个请求长时间卡住
                if result:
                    results.append(result)
            except TimeoutError:
                idx = futures[future]
                print(f"\n[Timeout] Sample {idx} exceeded 120s, skipping.")
            except Exception as e:
                idx = futures[future]
                print(f"\n[Error] Sample {idx} failed: {e}")

    # 5. Save Output
    print(f"\nSuccessfully generated {len(results)} samples.")
    print(f"Saving to: {OUTPUT_DATASET}")
    
    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DATASET, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n[Preview of one generated sample output]:")
    print("-" * 40)
    if results:
        print(results[0]['output'])
    print("-" * 40)
    print("Done! You can now use this file for Step 2 fine-tuning.")

if __name__ == "__main__":
    main()