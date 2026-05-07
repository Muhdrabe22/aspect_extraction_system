"""
Module 2: Multi-Domain Feature Weighting
Implements domain-aware TF-IDF feature weighting with:
- Domain-specific aspect seed boosting
- Cross-domain shared feature detection
- Per-token weight scores used by the model
- Feature importance visualization support
"""

import os
import sys
import math
import json
import pickle
import numpy as np
from collections import defaultdict, Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# ─────────────────────────────────────────────
# 1. Domain-Aware TF-IDF Engine
# ─────────────────────────────────────────────

class DomainTFIDF:
    """
    Computes TF-IDF scores per domain separately.
    Enables domain-specific vocabulary weighting.
    """

    def __init__(self):
        self.domain_doc_freq = {}          # domain -> {term -> doc_freq}  (plain dict, picklable)
        self.domain_doc_count = defaultdict(int)
        self.global_doc_freq = defaultdict(int)
        self.global_doc_count = 0
        self.fitted = False

    def _tokenize(self, text):
        """Simple lowercase word tokenizer."""
        import re
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def fit(self, dataset):
        """
        Fit TF-IDF on multi-domain dataset.
        dataset: dict {domain: [records]}  where record has 'text' field
        """
        for domain, records in dataset.items():
            if domain not in self.domain_doc_freq:
                self.domain_doc_freq[domain] = {}
            for record in records:
                tokens = set(self._tokenize(record['text']))
                for token in tokens:
                    self.domain_doc_freq[domain][token] = self.domain_doc_freq[domain].get(token, 0) + 1
                    self.global_doc_freq[token] += 1
                self.domain_doc_count[domain] += 1
                self.global_doc_count += 1

        self.fitted = True
        print(f"[TFIDF] Fitted on {self.global_doc_count} docs across {len(dataset)} domains")
        return self

    def domain_idf(self, term, domain):
        df = self.domain_doc_freq.get(domain, {}).get(term, 0)
        n = self.domain_doc_count.get(domain, 1)
        return math.log((n + 1) / (df + 1)) + 1

    def global_idf(self, term):
        """IDF score across all domains."""
        df = self.global_doc_freq.get(term, 0)
        return math.log((self.global_doc_count + 1) / (df + 1)) + 1

    def tf(self, term, tokens):
        """Term frequency in a token list."""
        count = tokens.count(term)
        return count / len(tokens) if tokens else 0

    def score(self, term, tokens, domain):
        """Combined domain-aware TF-IDF score."""
        tf = self.tf(term, tokens)
        d_idf = self.domain_idf(term, domain)
        return tf * d_idf

    def top_terms(self, domain, topk=20):
        scores = {
            term: self.domain_idf(term, domain)
            for term in self.domain_doc_freq.get(domain, {})
        }
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]


# ─────────────────────────────────────────────
# 2. Aspect Seed Booster
# ─────────────────────────────────────────────

class AspectSeedBooster:
    """
    Boosts weights of tokens that match or are similar to
    known domain aspect seeds. Provides prior knowledge injection.
    """

    def __init__(self, domain_seeds=None):
        self.domain_seeds = domain_seeds or Config.DOMAIN_ASPECT_SEEDS
        # Build lowercase seed sets
        self.seed_sets = {
            domain: set(s.lower() for s in seeds)
            for domain, seeds in self.domain_seeds.items()
        }
        # Multi-word seeds (ngrams)
        self.ngram_seeds = {
            domain: [s.lower() for s in seeds if ' ' in s]
            for domain, seeds in self.domain_seeds.items()
        }

    def boost_score(self, token, domain, base_score, boost_factor=2.5):
        """
        Apply boost if token matches a seed aspect.
        Returns boosted score.
        """
        token_lower = token.lower()
        seeds = self.seed_sets.get(domain, set())

        # Exact match
        if token_lower in seeds:
            return base_score * boost_factor

        # Partial match (token is part of a multi-word seed)
        for seed in seeds:
            if token_lower in seed.split():
                return base_score * (boost_factor * 0.7)

        return base_score

    def get_ngram_boosts(self, text, domain):
        """
        Returns character spans of ngram seed matches in text.
        Used for phrase-level boosting.
        """
        import re
        text_lower = text.lower()
        matches = []
        for ngram in self.ngram_seeds.get(domain, []):
            for m in re.finditer(re.escape(ngram), text_lower):
                matches.append({
                    'ngram': ngram,
                    'start': m.start(),
                    'end': m.end()
                })
        return matches

    def is_seed_token(self, token, domain):
        return token.lower() in self.seed_sets.get(domain, set())


# ─────────────────────────────────────────────
# 3. Cross-Domain Shared Feature Detector
# ─────────────────────────────────────────────

class CrossDomainFeatureDetector:
    """
    Identifies features that are shared across domains
    (universal aspects like 'price', 'quality', 'staff')
    vs domain-specific features.

    Shared features get a transfer bonus during prediction.
    """

    def __init__(self, tfidf: DomainTFIDF, domains=None):
        self.tfidf = tfidf
        self.domains = domains or Config.DOMAINS
        self.shared_features = set()
        self.domain_specific = defaultdict(set)
        self.transferability_scores = {}

    def compute(self, min_domains=2, top_per_domain=50):
        """
        Identify shared vs domain-specific terms.
        A term is 'shared' if it appears as top-term in >= min_domains domains.
        """
        domain_top_terms = {}
        for domain in self.domains:
            top = self.tfidf.top_terms(domain, topk=top_per_domain)
            domain_top_terms[domain] = set(t for t, _ in top)

        # Count how many domains each term appears in
        term_domain_count = defaultdict(int)
        term_domains = defaultdict(set)
        for domain, terms in domain_top_terms.items():
            for term in terms:
                term_domain_count[term] += 1
                term_domains[term].add(domain)

        # Classify
        for term, count in term_domain_count.items():
            if count >= min_domains:
                self.shared_features.add(term)
                self.transferability_scores[term] = count / len(self.domains)
            else:
                for domain in term_domains[term]:
                    self.domain_specific[domain].add(term)

        print(f"[CROSS-DOMAIN] Shared features: {len(self.shared_features)}")
        for domain in self.domains:
            print(f"[CROSS-DOMAIN] {domain}-specific: {len(self.domain_specific[domain])}")

        return self

    def transfer_bonus(self, token, source_domain, target_domain, bonus=1.3):
        """
        Returns transfer bonus multiplier for a token.
        Shared features get a bonus when transferring across domains.
        """
        if token.lower() in self.shared_features:
            return bonus
        return 1.0

    def is_shared(self, token):
        return token.lower() in self.shared_features


# ─────────────────────────────────────────────
# 4. Feature Weight Computer (Main Class)
# ─────────────────────────────────────────────

class FeatureWeightComputer:
    """
    Main feature weighting engine.
    Combines TF-IDF, seed boosting, and cross-domain transfer
    into per-token weight vectors used by the aspect extraction model.
    """

    def __init__(self):
        self.tfidf = DomainTFIDF()
        self.booster = AspectSeedBooster()
        self.cross_domain = None
        self.fitted = False

    def fit(self, dataset):
        """Fit all components on dataset."""
        print("[WEIGHTS] Fitting feature weight computer...")

        # Fit TF-IDF
        self.tfidf.fit(dataset)

        # Compute cross-domain features
        self.cross_domain = CrossDomainFeatureDetector(self.tfidf)
        self.cross_domain.compute()

        self.fitted = True
        print("[WEIGHTS] ✅ Feature weight computer fitted")
        return self

    def compute_token_weights(self, tokens, domain, source_domain=None):
        """
        Compute per-token weight scores for a list of tokens.

        Returns: list of floats, one per token
        """
        if not self.fitted:
            raise RuntimeError("FeatureWeightComputer not fitted. Call fit() first.")

        tokens_lower = [t.lower() for t in tokens]
        weights = []

        for token in tokens_lower:
            # Base TF-IDF score
            base = self.tfidf.score(token, tokens_lower, domain)

            # Seed boost
            boosted = self.booster.boost_score(token, domain, base)

            # Cross-domain transfer bonus
            if source_domain and source_domain != domain:
                transfer = self.cross_domain.transfer_bonus(
                    token, source_domain, domain
                )
                boosted *= transfer

            weights.append(boosted)

        # Normalize to [0, 1]
        max_w = max(weights) if weights else 1.0
        if max_w > 0:
            weights = [w / max_w for w in weights]

        return weights

    def compute_record_weights(self, record, source_domain=None):
        """
        Compute weights for a full record.
        Returns record with added 'token_weights' field.
        """
        tokens = record.get('tokens', [])
        domain = record.get('domain', 'restaurants')

        weights = self.compute_token_weights(tokens, domain, source_domain)

        return {**record, 'token_weights': weights}

    def weight_dataset(self, dataset, source_domain=None):
        """
        Add token weights to all records in dataset.
        Returns weighted dataset.
        """
        weighted = defaultdict(list)
        total = 0

        for domain, records in dataset.items():
            for record in records:
                weighted_record = self.compute_record_weights(record, source_domain)
                weighted[domain].append(weighted_record)
                total += 1

        print(f"[WEIGHTS] Computed weights for {total} records")
        return dict(weighted)

    def get_domain_feature_report(self, domain):
        """
        Generate a feature importance report for a domain.
        Returns dict with top features, seeds, shared features.
        """
        top_terms = self.tfidf.top_terms(domain, topk=15)
        seeds = list(self.booster.seed_sets.get(domain, set()))[:10]
        shared = list(self.cross_domain.shared_features)[:10]
        specific = list(self.cross_domain.domain_specific.get(domain, set()))[:10]

        return {
            'domain': domain,
            'top_tfidf_terms': [{'term': t, 'score': round(s, 4)} for t, s in top_terms],
            'seed_aspects': sorted(seeds),
            'shared_features': sorted(shared),
            'domain_specific_features': sorted(specific)
        }

    def save(self, path=None):
        """Save fitted feature weight computer."""
        if path is None:
            path = os.path.join(Config.MODEL_DIR, 'feature_weighter.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[WEIGHTS] Saved to {path}")

    @classmethod
    def load(cls, path=None):
        """Load fitted feature weight computer."""
        if path is None:
            path = os.path.join(Config.MODEL_DIR, 'feature_weighter.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved feature weighter at {path}")
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"[WEIGHTS] Loaded from {path}")
        return obj


# ─────────────────────────────────────────────
# 5. Lightweight Inference (no BERT needed)
# ─────────────────────────────────────────────

def compute_weights_for_text(text, domain, weighter: FeatureWeightComputer):
    """
    Convenience function: compute token weights for raw text.
    Returns: (tokens, weights)
    """
    import re
    tokens = re.findall(r'\S+', text)
    weights = weighter.compute_token_weights(tokens, domain)
    return tokens, weights


# ─────────────────────────────────────────────
# 6. Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("Module 2: Multi-Domain Feature Weighting")
    print("=" * 55)

    # Load dataset from Module 1
    from ml.dataset import load_processed, build_dataset, split_dataset, save_processed

    try:
        dataset, splits = load_processed()
        print("[LOAD] Loaded existing dataset")
    except FileNotFoundError:
        print("[LOAD] Building dataset fresh...")
        dataset, stats = build_dataset()
        splits = split_dataset(dataset)
        save_processed(dataset, splits)

    # Fit feature weight computer
    weighter = FeatureWeightComputer()
    weighter.fit(dataset)

    # Add weights to dataset
    weighted_dataset = weighter.weight_dataset(dataset)

    # Domain feature reports
    print("\n[REPORT] Domain Feature Analysis:")
    for domain in Config.DOMAINS:
        report = weighter.get_domain_feature_report(domain)
        print(f"\n  ── {domain.upper()} ──")
        print(f"  Top TF-IDF: {[t['term'] for t in report['top_tfidf_terms'][:8]]}")
        print(f"  Seed Aspects: {report['seed_aspects'][:6]}")
        print(f"  Shared: {report['shared_features'][:5]}")
        print(f"  Domain-Specific: {report['domain_specific_features'][:5]}")

    # Test on a sample sentence
    print("\n[TEST] Token Weights on Sample Reviews:")
    samples = [
        ("The battery life is amazing but the screen is too dim.", "electronics"),
        ("Service was slow but the food tasted incredible.", "restaurants"),
        ("Room was clean and the staff were very helpful.", "hotels"),
    ]
    for text, domain in samples:
        tokens, weights = compute_weights_for_text(text, domain, weighter)
        token_scores = sorted(zip(tokens, weights), key=lambda x: -x[1])[:5]
        print(f"\n  [{domain}] \"{text[:50]}...\"")
        print(f"  Top weighted tokens: {[(t, round(w, 3)) for t, w in token_scores]}")

    # Save
    weighter.save()

    # Save weighted dataset
    weighted_path = os.path.join(Config.DATA_DIR, 'weighted_dataset.pkl')
    with open(weighted_path, 'wb') as f:
        pickle.dump(weighted_dataset, f)
    print(f"\n[SAVE] Weighted dataset saved to {weighted_path}")

    print("\n✅ Module 2 complete.")
