#!/usr/bin/env python3
"""从原始网页目录生成 ## URL / ## Content / ## External Links 格式 JSON。仅供学术研究，禁止用于非法用途。"""

from __future__ import annotations

import ast
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import os

from bs4 import BeautifulSoup
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
PHISHING_DIR = Path(os.environ.get("RAW_PHISHING_DIR", REPO_ROOT / "data/raw/temp_phishing"))
LEGIT_DIR = Path(os.environ.get("RAW_LEGIT_DIR", REPO_ROOT / "data/raw/temp_normal"))
OUTPUT_PATH = REPO_ROOT / "data" / "dataset_scamnet_5000.json"

SAMPLE_COUNT = 4000     # Number of samples per class
HTML_TOKEN_LIMIT = 2000 # Character limit for HTML content
LINKS_LIMIT = 20        # Max number of external links to record

def iter_subdirs(parent: Path) -> Iterable[Path]:
    """Iterate over subdirectories."""
    if not parent.exists():
        return []
    for child in sorted(parent.iterdir()):
        if child.is_dir():
            yield child

def is_valid_url(url: str) -> bool:
    """Basic validation for URLs."""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    if len(url) > 200 or "(" in url or "<" in url:
        return False
    if url.startswith(("http://", "https://", "www.")):
        return True
    return False

def folder_name_to_url(folder: Path) -> Optional[str]:
    """Convert folder name to URL for legit dataset."""
    url = folder.name.replace("+", "/")
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url if is_valid_url(url) else None

def phishing_info_url(site_dir: Path) -> Optional[str]:
    """Extract URL from info.txt for phishing dataset."""
    info_path = site_dir / "info.txt"
    if not info_path.is_file():
        return None
    try:
        content = info_path.read_text(encoding="utf-8").strip()
        data = ast.literal_eval(content)
        if isinstance(data, dict):
            url = data.get("url")
            if url and is_valid_url(url):
                return url.strip()
    except Exception:
        pass
    return None

def extract_html_features(site_dir: Path) -> Tuple[str, str]:
    """
    Parses html.txt to extract:
    1. Visible Body Text (Content)
    2. List of External Links
    """
    html_path = site_dir / "html.txt"
    if not html_path.is_file():
        return "", ""

    try:
        content = html_path.read_text(encoding="utf-8", errors="ignore")
        if not content:
            return "", ""

        soup = BeautifulSoup(content, 'html.parser')

        # 1. Extract External Links
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            # Basic filter for external links
            if href.startswith(('http://', 'https://')):
                links.add(href)
        
        # Format links list (truncate if too many)
        links_list = list(links)[:LINKS_LIMIT]
        links_str = str(links_list) if links_list else "None"

        # 2. Extract Visible Text (Content)
        # Remove scripts, styles, meta tags
        for script in soup(["script", "style", "meta", "noscript", "iframe"]):
            script.extract()
        
        text = soup.get_text(separator=' ', strip=True)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate content to fit context window
        text = text[:HTML_TOKEN_LIMIT]
        
        return text, links_str

    except Exception as e:
        return "", "None"

def process_single_sample(
    site_dir: Path,
    label_text: str, # "legit" or "scam"
    is_phishing: bool,
) -> Optional[dict]:
    
    # 1. Get URL
    if is_phishing:
        url = phishing_info_url(site_dir)
    else:
        url = folder_name_to_url(site_dir)
    
    if not url:
        return None

    # 2. Parse HTML for Content and Links
    content, external_links = extract_html_features(site_dir)
    if not content or len(content) < 50: # Skip if content is empty or too short
        return None

    input_text = (
        f"# Information:\n"
        f"## URL:\n{url}\n"
        f"## Content:\n{content}\n"
        f"## External Links:\n{external_links}\n"
    )

    sample = {
        "input": input_text,
        "output": f"Label: {label_text}", 
    }
    return sample

def gather_samples(
    directory: Path,
    label_text: str,
    is_phishing: bool = False,
    sample_count: int = SAMPLE_COUNT,
) -> List[dict]:
    
    print(f"\nProcessing {label_text} data from: {directory}")
    all_subdirs = list(iter_subdirs(directory))
    
    if len(all_subdirs) == 0:
        print(f"  [WARNING] No subdirectories found in {directory}")
        return []

    if len(all_subdirs) > sample_count:
        selected_dirs = random.sample(all_subdirs, sample_count)
        print(f"  Selected {sample_count} random samples from {len(all_subdirs)}")
    else:
        selected_dirs = all_subdirs
        print(f"  Processing all {len(all_subdirs)} samples")

    samples = []
    
    for site_dir in tqdm(selected_dirs, desc=f"  Building {label_text}", unit="sample"):
        sample = process_single_sample(site_dir, label_text, is_phishing)
        if sample:
            samples.append(sample)
            
    return samples

def main():
    print("=" * 60)
    print("ScamNet Dataset Generator (Content-Only Variant)")
    print("=" * 60)

    # Validate Directories
    if not PHISHING_DIR.exists():
        print(f"[ERROR] Phishing directory not found: {PHISHING_DIR}")
        sys.exit(1)
    if not LEGIT_DIR.exists():
        print(f"[ERROR] Legit directory not found: {LEGIT_DIR}")
        sys.exit(1)

    dataset = []

    # Process Phishing (Label: scam)
    phishing_data = gather_samples(PHISHING_DIR, "scam", is_phishing=True)
    dataset.extend(phishing_data)

    # Process Legit (Label: legit)
    legit_data = gather_samples(LEGIT_DIR, "legit", is_phishing=False)
    dataset.extend(legit_data)

    # Shuffle
    random.shuffle(dataset)

    # Check balance
    p_count = len(phishing_data)
    l_count = len(legit_data)
    print(f"\nStats: Scam: {p_count}, Legit: {l_count}, Total: {len(dataset)}")

    if len(dataset) == 0:
        print("[ERROR] No valid samples generated. Check your directories and input files.")
        sys.exit(1)

    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    try:
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print("Done! Dataset ready for training.")
    except Exception as e:
        print(f"[ERROR] Failed to save JSON: {e}")

if __name__ == "__main__":
    main()