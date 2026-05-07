"""
Module 1: Data Pipeline
Parses SemEval 2014 XML format, builds training datasets,
handles multi-domain feature extraction and BIO tagging.
"""

import os
import re
import xml.etree.ElementTree as ET
import json
import pickle
from collections import defaultdict
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# ─────────────────────────────────────────────
# 1. SemEval XML Parser
# ─────────────────────────────────────────────

def parse_semeval_xml(filepath, domain='restaurants'):
    """
    Parses SemEval 2014 Task 4 XML files.
    Returns list of dicts: {text, aspects: [{term, polarity, from, to}]}
    """
    records = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        for sentence in root.findall('.//sentence'):
            text_el = sentence.find('text')
            if text_el is None or text_el.text is None:
                continue

            text = text_el.text.strip()
            aspects = []

            aspect_terms = sentence.find('aspectTerms')
            if aspect_terms is not None:
                for at in aspect_terms.findall('aspectTerm'):
                    term = at.get('term', '').strip()
                    polarity = at.get('polarity', 'neutral').strip()
                    from_idx = int(at.get('from', 0))
                    to_idx = int(at.get('to', 0))

                    if term and term.lower() != 'null':
                        aspects.append({
                            'term': term,
                            'polarity': polarity,
                            'from': from_idx,
                            'to': to_idx
                        })

            records.append({
                'text': text,
                'aspects': aspects,
                'domain': domain
            })

    except ET.ParseError as e:
        print(f"[ERROR] Failed to parse {filepath}: {e}")

    return records


# ─────────────────────────────────────────────
# 2. BIO Tagger
# ─────────────────────────────────────────────

def tokenize_and_bio_tag(record):
    """
    Converts a record into word tokens + BIO labels + sentiment labels.
    B-ASP = Beginning of aspect, I-ASP = Inside aspect, O = Outside
    """
    text = record['text']
    aspects = record['aspects']

    # Simple whitespace tokenizer with char offset tracking
    tokens = []
    offsets = []
    i = 0
    for match in re.finditer(r'\S+', text):
        tokens.append(match.group())
        offsets.append((match.start(), match.end()))

    # Build aspect span map: char_idx -> (label, polarity)
    aspect_map = {}
    for asp in aspects:
        for c in range(asp['from'], asp['to']):
            aspect_map[c] = asp

    # Assign BIO labels
    bio_labels = []
    sentiment_labels = []

    for token, (start, end) in zip(tokens, offsets):
        token_chars = set(range(start, end))
        overlapping = [aspect_map[c] for c in token_chars if c in aspect_map]

        if not overlapping:
            bio_labels.append('O')
            sentiment_labels.append('neutral')
        else:
            asp = overlapping[0]
            if start == asp['from'] or not any(
                c in aspect_map and aspect_map[c] == asp
                for c in range(start - 1, start)
            ):
                bio_labels.append('B-ASP')
            else:
                bio_labels.append('I-ASP')
            sentiment_labels.append(asp['polarity'])

    return {
        'text': text,
        'tokens': tokens,
        'bio_labels': bio_labels,
        'sentiment_labels': sentiment_labels,
        'domain': record['domain']
    }


# ─────────────────────────────────────────────
# 3. Synthetic Data Generator (for Hotels + fallback)
# ─────────────────────────────────────────────

SYNTHETIC_REVIEWS = {
    'hotels': [
        ("The room was spacious and very clean.", [('room', 'positive', 4, 8), ('clean', 'positive', 30, 35)]),
        ("Staff were rude and unhelpful at check-in.", [('Staff', 'negative', 0, 5), ('check-in', 'negative', 33, 41)]),
        ("Great location but the wifi was terrible.", [('location', 'positive', 6, 14), ('wifi', 'negative', 23, 27)]),
        ("The bed was comfortable and the view was stunning.", [('bed', 'positive', 4, 7), ('view', 'positive', 34, 38)]),
        ("Breakfast was excellent with lots of variety.", [('Breakfast', 'positive', 0, 9)]),
        ("Bathroom was dirty and the towels smelled bad.", [('Bathroom', 'negative', 0, 8), ('towels', 'negative', 28, 34)]),
        ("Parking is free and very convenient.", [('Parking', 'positive', 0, 7)]),
        ("The pool was closed during our entire stay.", [('pool', 'negative', 4, 8)]),
        ("Noise from the street made it hard to sleep.", [('Noise', 'negative', 0, 5)]),
        ("Check-in was smooth and the receptionist was friendly.", [('Check-in', 'positive', 0, 8), ('receptionist', 'positive', 29, 41)]),
        ("The amenities were outdated but the price was fair.", [('amenities', 'negative', 4, 13), ('price', 'positive', 36, 41)]),
        ("Room service was prompt and the food quality was good.", [('Room service', 'positive', 0, 12), ('food quality', 'positive', 33, 45)]),
    ],
    'electronics': [
        ("The battery life is excellent, lasts all day.", [('battery life', 'positive', 4, 16)]),
        ("Camera takes stunning photos even in low light.", [('Camera', 'positive', 0, 6)]),
        ("The screen is too dim for outdoor use.", [('screen', 'negative', 4, 10)]),
        ("Performance is blazing fast with no lag at all.", [('Performance', 'positive', 0, 11)]),
        ("Build quality feels cheap and plasticky.", [('Build quality', 'negative', 0, 13)]),
        ("The keyboard is comfortable for long typing sessions.", [('keyboard', 'positive', 4, 12)]),
        ("Speaker output is surprisingly loud and clear.", [('Speaker', 'positive', 0, 7)]),
        ("Charging speed is disappointingly slow.", [('Charging speed', 'negative', 0, 14)]),
        ("Storage space runs out quickly.", [('Storage space', 'negative', 0, 13)]),
        ("Display colors are vivid and sharp.", [('Display', 'positive', 0, 7)]),
        ("The price is too high for what you get.", [('price', 'negative', 4, 9)]),
        ("Software updates are frequent and smooth.", [('Software updates', 'positive', 0, 16)]),
    ],
    'restaurants': [
        ("The food was delicious and portions were generous.", [('food', 'positive', 4, 8), ('portions', 'positive', 29, 37)]),
        ("Service was extremely slow on a busy night.", [('Service', 'negative', 0, 7)]),
        ("The ambiance is cozy and perfect for a date.", [('ambiance', 'positive', 4, 12)]),
        ("Staff were attentive and very knowledgeable.", [('Staff', 'positive', 0, 5)]),
        ("Prices are quite reasonable for the quality.", [('Prices', 'positive', 0, 6)]),
        ("The menu has plenty of vegetarian options.", [('menu', 'positive', 4, 8)]),
        ("Wait time was over an hour without any updates.", [('Wait time', 'negative', 0, 9)]),
        ("Dessert was heavenly — best cheesecake ever.", [('Dessert', 'positive', 0, 7)]),
        ("Location is hard to find with no clear signage.", [('Location', 'negative', 0, 8)]),
        ("Drinks were overpriced and poorly mixed.", [('Drinks', 'negative', 0, 6)]),
        ("Cleanliness was impeccable throughout.", [('Cleanliness', 'positive', 0, 11)]),
        ("Taste of the pasta was bland and unseasoned.", [('Taste', 'negative', 0, 5), ('pasta', 'negative', 13, 18)]),
    ]
}


def generate_synthetic_records(domain):
    """Generate synthetic labeled records for a domain."""
    records = []
    if domain not in SYNTHETIC_REVIEWS:
        return records

    for text, aspect_list in SYNTHETIC_REVIEWS[domain]:
        aspects = []
        for term, polarity, from_idx, to_idx in aspect_list:
            aspects.append({
                'term': term,
                'polarity': polarity,
                'from': from_idx,
                'to': to_idx
            })
        records.append({'text': text, 'aspects': aspects, 'domain': domain})

    return records


# ─────────────────────────────────────────────
# 4. Full Dataset Builder
# ─────────────────────────────────────────────

def build_dataset(data_dir=None, domains=None):
    """
    Build training/validation datasets from SemEval XML files.
    Falls back to synthetic data if XML files not found.
    Returns dict: {domain: [tagged_records]}
    """
    if data_dir is None:
        data_dir = Config.DATA_DIR
    if domains is None:
        domains = Config.DOMAINS

    dataset = defaultdict(list)
    stats = {}

    # Known SemEval filenames
    semeval_files = {
        'restaurants': [
            'Restaurants_Train_v2.xml',
            'restaurants_train.xml',
            'restaurant_train.xml',
        ],
        'electronics': [
            'Laptop_Train_v2.xml',
            'laptops_train.xml',
            'laptop_train.xml',
        ]
    }

    for domain in domains:
        raw_records = []

        # Try to load SemEval XML
        if domain in semeval_files:
            for fname in semeval_files[domain]:
                fpath = os.path.join(data_dir, fname)
                if os.path.exists(fpath):
                    print(f"[DATA] Loading {domain} from {fname}...")
                    raw_records = parse_semeval_xml(fpath, domain)
                    break

        # Supplement or replace with synthetic data
        synthetic = generate_synthetic_records(domain)
        if not raw_records:
            print(f"[DATA] No XML found for '{domain}', using synthetic data ({len(synthetic)} records)")
            raw_records = synthetic
        else:
            print(f"[DATA] Loaded {len(raw_records)} XML records for '{domain}', adding {len(synthetic)} synthetic")
            raw_records += synthetic

        # Apply BIO tagging
        tagged = []
        for rec in raw_records:
            try:
                tagged.append(tokenize_and_bio_tag(rec))
            except Exception as e:
                print(f"[WARN] Skipping record: {e}")

        dataset[domain] = tagged
        stats[domain] = {
            'total': len(tagged),
            'with_aspects': sum(1 for r in tagged if 'B-ASP' in r['bio_labels']),
            'total_aspects': sum(r['bio_labels'].count('B-ASP') for r in tagged)
        }
        print(f"[DATA] {domain}: {stats[domain]['total']} records, "
              f"{stats[domain]['with_aspects']} with aspects, "
              f"{stats[domain]['total_aspects']} total aspects")

    return dict(dataset), stats


# ─────────────────────────────────────────────
# 5. Train/Val Split
# ─────────────────────────────────────────────

def split_dataset(dataset, val_ratio=0.15):
    """Split each domain into train/val sets."""
    splits = {}
    for domain, records in dataset.items():
        n = len(records)
        val_size = max(1, int(n * val_ratio))
        # Simple sequential split (shuffle handled in DataLoader)
        splits[domain] = {
            'train': records[val_size:],
            'val': records[:val_size]
        }
        print(f"[SPLIT] {domain}: {len(splits[domain]['train'])} train, "
              f"{len(splits[domain]['val'])} val")
    return splits


# ─────────────────────────────────────────────
# 6. Save / Load Processed Data
# ─────────────────────────────────────────────

def save_processed(dataset, splits, output_dir=None):
    """Save processed dataset to disk."""
    if output_dir is None:
        output_dir = Config.DATA_DIR
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)

    # Also save as JSON for inspection
    with open(os.path.join(output_dir, 'dataset_sample.json'), 'w') as f:
        sample = {d: records[:3] for d, records in dataset.items()}
        json.dump(sample, f, indent=2)

    print(f"[SAVE] Dataset saved to {output_dir}")


def load_processed(data_dir=None):
    """Load processed dataset from disk."""
    if data_dir is None:
        data_dir = Config.DATA_DIR

    dataset_path = os.path.join(data_dir, 'dataset.pkl')
    splits_path = os.path.join(data_dir, 'splits.pkl')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No processed dataset found at {dataset_path}. Run build_dataset() first.")

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)

    return dataset, splits


# ─────────────────────────────────────────────
# 7. Label Encoders
# ─────────────────────────────────────────────

BIO_LABEL2ID = {'O': 0, 'B-ASP': 1, 'I-ASP': 2}
BIO_ID2LABEL = {v: k for k, v in BIO_LABEL2ID.items()}

SENTIMENT_LABEL2ID = {'positive': 0, 'negative': 1, 'neutral': 2, 'conflict': 3}
SENTIMENT_ID2LABEL = {v: k for k, v in SENTIMENT_LABEL2ID.items()}

DOMAIN_LABEL2ID = {'electronics': 0, 'restaurants': 1, 'hotels': 2}
DOMAIN_ID2LABEL = {v: k for k, v in DOMAIN_LABEL2ID.items()}


# ─────────────────────────────────────────────
# 8. Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("Module 1: Data Pipeline")
    print("=" * 50)

    dataset, stats = build_dataset()
    splits = split_dataset(dataset)
    save_processed(dataset, splits)

    print("\n[STATS] Final Dataset Summary:")
    for domain, s in stats.items():
        print(f"  {domain:15s}: {s['total']:4d} records | "
              f"{s['with_aspects']:4d} with aspects | "
              f"{s['total_aspects']:4d} aspects total")

    print("\n✅ Module 1 complete.")
