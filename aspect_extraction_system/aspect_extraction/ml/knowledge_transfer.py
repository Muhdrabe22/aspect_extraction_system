"""
Module 5: Knowledge Transfer Layer
Cross-domain knowledge transfer for aspect extraction.

Techniques:
1. Shared Encoder   - common feature space across domains
2. Domain Adapter   - lightweight domain-specific adapter layers
3. Pivot Features   - shared vocabulary bridge between domains
4. Instance Weighting - re-weight source domain samples by similarity
5. Aspect Taxonomy  - hierarchical aspect mapping across domains
"""

import os
import sys
import re
import json
import pickle
import math
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from ml.feature_weighting import FeatureWeightComputer


# ─────────────────────────────────────────────
# 1. Aspect Taxonomy (Cross-Domain Mapping)
# ─────────────────────────────────────────────

class AspectTaxonomy:
    """
    Hierarchical aspect taxonomy that maps domain-specific
    aspect terms to universal categories.

    Universal categories: quality, price, service, ambiance,
                          comfort, performance, cleanliness, staff
    """

    TAXONOMY = {
        'quality': {
            'electronics':   ['build quality', 'material', 'durability', 'design', 'finish'],
            'restaurants':   ['food quality', 'freshness', 'taste', 'presentation', 'ingredients'],
            'hotels':        ['room quality', 'furnishing', 'maintenance', 'cleanliness', 'condition'],
        },
        'performance': {
            'electronics':   ['speed', 'performance', 'processor', 'battery', 'storage', 'memory', 'charging'],
            'restaurants':   ['service speed', 'wait time', 'efficiency', 'delivery'],
            'hotels':        ['check-in speed', 'service speed', 'elevator', 'wifi speed'],
        },
        'price': {
            'electronics':   ['price', 'cost', 'value', 'affordability'],
            'restaurants':   ['price', 'cost', 'value', 'bill', 'portion'],
            'hotels':        ['price', 'rate', 'cost', 'value for money'],
        },
        'staff': {
            'electronics':   ['support', 'customer service', 'helpline', 'repair service'],
            'restaurants':   ['staff', 'waiter', 'server', 'manager', 'chef', 'service'],
            'hotels':        ['staff', 'receptionist', 'concierge', 'housekeeping', 'management'],
        },
        'ambiance': {
            'electronics':   ['design', 'aesthetics', 'look', 'color', 'style'],
            'restaurants':   ['ambiance', 'atmosphere', 'decor', 'music', 'lighting', 'seating'],
            'hotels':        ['lobby', 'decor', 'atmosphere', 'view', 'garden', 'pool area'],
        },
        'comfort': {
            'electronics':   ['ergonomics', 'keyboard feel', 'weight', 'grip', 'screen size'],
            'restaurants':   ['seating', 'comfort', 'space', 'temperature', 'noise level'],
            'hotels':        ['bed', 'pillow', 'mattress', 'room size', 'bathroom', 'shower'],
        },
        'connectivity': {
            'electronics':   ['wifi', 'bluetooth', 'usb', 'ports', 'network', '5g', '4g'],
            'restaurants':   ['location', 'accessibility', 'parking', 'transport'],
            'hotels':        ['wifi', 'location', 'transport', 'parking', 'accessibility'],
        },
        'cleanliness': {
            'electronics':   ['screen cleanliness', 'dust', 'smudge'],
            'restaurants':   ['cleanliness', 'hygiene', 'sanitation', 'kitchen', 'toilets'],
            'hotels':        ['cleanliness', 'housekeeping', 'towels', 'sheets', 'bathroom'],
        },
    }

    def __init__(self):
        self.domain_term_cat = {}   # (term, domain) -> category
        self.global_term_cat = {}   # term -> category
        for category, domain_map in self.TAXONOMY.items():
            for domain, terms in domain_map.items():
                for term in terms:
                    self.domain_term_cat[(term.lower(), domain)] = category
                    if term.lower() not in self.global_term_cat:
                        self.global_term_cat[term.lower()] = category

    def get_category(self, aspect_term, domain=None):
        """Map aspect term to universal category."""
        term = aspect_term.lower()
        if domain:
            if (term, domain) in self.domain_term_cat:
                return self.domain_term_cat[(term, domain)]
            for (t, d), cat in self.domain_term_cat.items():
                if d == domain and (term in t or t in term):
                    return cat
        if term in self.global_term_cat:
            return self.global_term_cat[term]
        for t, cat in self.global_term_cat.items():
            if term in t or t in term:
                return cat
        return 'other'

    def get_domain_equivalents(self, aspect_term, source_domain, target_domain):
        """
        Find equivalent aspect terms in target domain.
        e.g., 'battery' (electronics) → 'service speed' (restaurants)
        """
        category = self.get_category(aspect_term, source_domain)
        if category == 'other':
            return []
        return self.TAXONOMY.get(category, {}).get(target_domain, [])

    def get_all_categories(self):
        return list(self.TAXONOMY.keys())


# ─────────────────────────────────────────────
# 2. Pivot Feature Bridge
# ─────────────────────────────────────────────

class PivotFeatureBridge:
    """
    Identifies pivot features — words that appear in both
    source and target domains with similar sentiment patterns.

    These pivots act as transfer anchors.
    """

    def __init__(self, weighter: FeatureWeightComputer = None):
        self.weighter = weighter
        self.pivots = {}           # term -> {domains, avg_weight, category}
        self.domain_pivots = defaultdict(set)  # domain_pair -> pivot_set

    def compute_pivots(self, dataset, min_domains=2, top_k=30):
        """
        Find pivot features shared across domains.
        dataset: {domain: [records]}
        """
        # Count term frequency per domain
        term_domain_freq = defaultdict(dict)
        term_domain_weight = defaultdict(dict)

        for domain, records in dataset.items():
            for record in records:
                tokens = re.findall(r'\b[a-z]{3,}\b', record['text'].lower())
                weights = record.get('token_weights', [1.0] * len(tokens))

                for i, tok in enumerate(tokens):
                    if tok not in term_domain_freq: term_domain_freq[tok] = {}
                    term_domain_freq[tok][domain] = term_domain_freq[tok].get(domain, 0) + 1
                    w = weights[i] if i < len(weights) else 1.0
                    if tok not in term_domain_weight: term_domain_weight[tok] = {}
                    if domain not in term_domain_weight[tok]: term_domain_weight[tok][domain] = []
                    term_domain_weight[tok][domain].append(w)

        # Select pivots: appear in >= min_domains
        for term, domain_freq in term_domain_freq.items():
            if len(domain_freq) >= min_domains:
                avg_weights = {
                    d: sum(ws) / len(ws)
                    for d, ws in term_domain_weight[term].items()
                }
                self.pivots[term] = {
                    'domains': list(domain_freq.keys()),
                    'freq': dict(domain_freq),
                    'avg_weight': avg_weights,
                    'pivot_score': sum(avg_weights.values()) / len(avg_weights)
                }

        # Build domain-pair pivot sets
        domains = list(dataset.keys())
        for i, d1 in enumerate(domains):
            for d2 in domains[i+1:]:
                pair_key = f"{d1}→{d2}"
                pair_pivots = {
                    t for t, info in self.pivots.items()
                    if d1 in info['domains'] and d2 in info['domains']
                }
                self.domain_pivots[pair_key] = pair_pivots
                self.domain_pivots[f"{d2}→{d1}"] = pair_pivots

        # Sort by pivot score, keep top_k
        sorted_pivots = sorted(
            self.pivots.items(),
            key=lambda x: x[1]['pivot_score'],
            reverse=True
        )[:top_k]
        self.pivots = dict(sorted_pivots)

        print(f"[TRANSFER] Found {len(self.pivots)} pivot features")
        for pair, pset in self.domain_pivots.items():
            print(f"[TRANSFER] Pivots {pair}: {len(pset)} shared terms")

        return self

    def get_pivots_for_pair(self, source, target):
        """Get pivot terms between two domains."""
        key = f"{source}→{target}"
        return self.domain_pivots.get(key, set())

    def pivot_weight(self, term, source_domain, target_domain):
        """
        Returns transfer weight for a term when moving
        from source to target domain.
        0 = no transfer value, 1 = full transfer value
        """
        if term not in self.pivots:
            return 0.0
        info = self.pivots[term]
        if source_domain not in info['domains'] or target_domain not in info['domains']:
            return 0.0
        # Weight by average importance across both domains
        s_w = info['avg_weight'].get(source_domain, 0)
        t_w = info['avg_weight'].get(target_domain, 0)
        return (s_w + t_w) / 2


# ─────────────────────────────────────────────
# 3. Instance Weighter
# ─────────────────────────────────────────────

class InstanceWeighter:
    """
    Re-weights source domain training instances by their
    similarity to the target domain.

    Higher-weight instances are more "transferable".
    Uses pivot feature overlap as similarity proxy.
    """

    def __init__(self, pivot_bridge: PivotFeatureBridge):
        self.bridge = pivot_bridge

    def compute_instance_weights(self, source_records, target_records,
                                  source_domain, target_domain):
        """
        Compute transfer weight for each source instance.
        Returns list of floats (one per source record).
        """
        # Build target domain vocabulary
        target_vocab = defaultdict(int)
        for rec in target_records:
            for tok in re.findall(r'\b[a-z]{3,}\b', rec['text'].lower()):
                target_vocab[tok] += 1

        # Get pivot set
        pivots = self.bridge.get_pivots_for_pair(source_domain, target_domain)

        weights = []
        for rec in source_records:
            tokens = set(re.findall(r'\b[a-z]{3,}\b', rec['text'].lower()))
            # Similarity = overlap with target vocab + pivot coverage
            vocab_overlap = sum(1 for t in tokens if t in target_vocab)
            pivot_overlap = sum(1 for t in tokens if t in pivots)

            sim = (vocab_overlap * 0.4 + pivot_overlap * 0.6) / max(len(tokens), 1)
            weights.append(round(min(sim * 5, 1.0), 4))  # scale to [0,1]

        return weights

    def get_top_transferable(self, source_records, target_records,
                              source_domain, target_domain, top_k=10):
        """Return top-k most transferable source instances."""
        weights = self.compute_instance_weights(
            source_records, target_records, source_domain, target_domain
        )
        ranked = sorted(
            zip(weights, source_records),
            key=lambda x: x[0], reverse=True
        )
        return [(w, r) for w, r in ranked[:top_k]]


# ─────────────────────────────────────────────
# 4. Domain Adapter
# ─────────────────────────────────────────────

class DomainAdapter:
    """
    Lightweight adapter that adjusts aspect predictions
    when transferring from source to target domain.

    Applies:
    - Taxonomy remapping (e.g., 'battery' → 'service speed')
    - Pivot-boosted confidence
    - Domain-specific threshold adjustment
    """

    def __init__(self, taxonomy: AspectTaxonomy,
                 pivot_bridge: PivotFeatureBridge,
                 weighter: FeatureWeightComputer = None):
        self.taxonomy = taxonomy
        self.bridge = pivot_bridge
        self.weighter = weighter

        # Confidence thresholds per domain (tuned empirically)
        self.thresholds = {
            'electronics': 0.55,
            'restaurants': 0.50,
            'hotels': 0.45,   # lower = more inclusive for sparse domain
        }

    def adapt_aspects(self, aspects, source_domain, target_domain, text):
        """
        Adapt extracted aspects from source domain for use in target domain.
        Returns list of adapted aspects with transfer metadata.
        """
        tokens = re.findall(r'\S+', text)
        adapted = []

        for asp in aspects:
            term = asp['term']

            # Get universal category
            category = self.taxonomy.get_category(term, source_domain)

            # Find target domain equivalents
            equivalents = self.taxonomy.get_domain_equivalents(
                term, source_domain, target_domain
            )

            # Compute pivot transfer weight
            term_lower = term.lower()
            pivot_w = self.bridge.pivot_weight(term_lower, source_domain, target_domain)

            # Apply transfer: if high pivot weight, include with bonus confidence
            threshold = self.thresholds.get(target_domain, 0.5)
            original_conf = asp.get('confidence', 0.5)
            transferred_conf = min(original_conf * (1 + pivot_w), 1.0)

            adapted.append({
                **asp,
                'source_domain': source_domain,
                'target_domain': target_domain,
                'category': category,
                'equivalents': equivalents[:3],
                'pivot_weight': round(pivot_w, 3),
                'transferred_confidence': round(transferred_conf, 3),
                'transferred': pivot_w > 0,
            })

        return adapted

    def filter_by_confidence(self, aspects, domain):
        """Filter aspects below domain confidence threshold."""
        threshold = self.thresholds.get(domain, 0.5)
        return [a for a in aspects if a.get('confidence', 1.0) >= threshold]


# ─────────────────────────────────────────────
# 5. Knowledge Transfer Engine (Main)
# ─────────────────────────────────────────────

class KnowledgeTransferEngine:
    """
    Main knowledge transfer coordinator.
    Orchestrates all transfer components.

    Usage:
      engine = KnowledgeTransferEngine()
      engine.fit(dataset)
      enriched = engine.transfer_predict(text, source='electronics', target='hotels')
    """

    def __init__(self, weighter: FeatureWeightComputer = None):
        self.weighter = weighter
        self.taxonomy = AspectTaxonomy()
        self.pivot_bridge = PivotFeatureBridge(weighter)
        self.instance_weighter = None
        self.adapter = None
        self.fitted = False
        self._transfer_stats = {}

    def fit(self, dataset):
        """Fit transfer components on multi-domain dataset."""
        print("[TRANSFER] Fitting knowledge transfer engine...")

        # Add weights to dataset if weighter available
        if self.weighter:
            for domain, records in dataset.items():
                for rec in records:
                    if 'token_weights' not in rec:
                        rec.update(self.weighter.compute_record_weights(rec))

        # Compute pivot features
        self.pivot_bridge.compute_pivots(dataset)

        # Instance weighter
        self.instance_weighter = InstanceWeighter(self.pivot_bridge)

        # Domain adapter
        self.adapter = DomainAdapter(
            self.taxonomy, self.pivot_bridge, self.weighter
        )

        self.fitted = True
        print("[TRANSFER] ✅ Knowledge transfer engine fitted")
        return self

    def enrich_aspects(self, aspects, domain):
        """
        Enrich aspects with taxonomy categories and transfer metadata.
        (Single-domain enrichment — no cross-domain transfer)
        """
        enriched = []
        for asp in aspects:
            category = self.taxonomy.get_category(asp['term'], domain)
            enriched.append({
                **asp,
                'category': category,
                'domain': domain,
            })
        return enriched

    def cross_domain_predict(self, text, source_domain, target_domain,
                              aspects_from_source):
        """
        Transfer aspect predictions from source domain to target domain.
        """
        if not self.fitted:
            return aspects_from_source

        adapted = self.adapter.adapt_aspects(
            aspects_from_source, source_domain, target_domain, text
        )
        key = f"{source_domain}_{target_domain}"
        self._transfer_stats[key] = self._transfer_stats.get(key, 0) + len(adapted)
        return adapted

    def get_transferable_knowledge(self, source_domain, target_domain, top_k=10):
        """
        Report on what knowledge can be transferred between domains.
        Returns structured report.
        """
        pivots = self.bridge_pivots(source_domain, target_domain)
        categories = []
        for cat in self.taxonomy.get_all_categories():
            src_terms = self.taxonomy.TAXONOMY.get(cat, {}).get(source_domain, [])
            tgt_terms = self.taxonomy.TAXONOMY.get(cat, {}).get(target_domain, [])
            if src_terms and tgt_terms:
                categories.append({
                    'category': cat,
                    'source_terms': src_terms[:3],
                    'target_terms': tgt_terms[:3],
                })
        return {
            'source': source_domain,
            'target': target_domain,
            'pivot_count': len(pivots),
            'top_pivots': list(pivots)[:top_k],
            'transferable_categories': categories,
        }

    def bridge_pivots(self, source, target):
        if not self.fitted:
            return set()
        return self.pivot_bridge.get_pivots_for_pair(source, target)

    def get_instance_transfer_weights(self, source_domain, target_domain, splits):
        """Get instance weights for cross-domain training."""
        if not self.fitted or not self.instance_weighter:
            return []
        source_recs = splits.get(source_domain, {}).get('train', [])
        target_recs = splits.get(target_domain, {}).get('train', [])
        return self.instance_weighter.compute_instance_weights(
            source_recs, target_recs, source_domain, target_domain
        )

    def save(self, path=None):
        if path is None:
            path = os.path.join(Config.MODEL_DIR, 'transfer_engine.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[TRANSFER] Saved to {path}")

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = os.path.join(Config.MODEL_DIR, 'transfer_engine.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("Module 5: Knowledge Transfer Layer")
    print("=" * 55)

    from ml.dataset import load_processed, build_dataset, split_dataset, save_processed
    from ml.feature_weighting import FeatureWeightComputer
    from ml.model import RuleBasedAspectModel
    from ml.sentiment import SentimentPipeline

    # Load data + weighter
    try:
        dataset, splits = load_processed()
    except FileNotFoundError:
        dataset, stats = build_dataset()
        splits = split_dataset(dataset)
        save_processed(dataset, splits)

    try:
        weighter = FeatureWeightComputer.load()
    except Exception:
        weighter = FeatureWeightComputer().fit(dataset)
        weighter.save()

    # Fit transfer engine
    engine = KnowledgeTransferEngine(weighter)
    engine.fit(dataset)

    # ── Test 1: Taxonomy lookup
    print("\n[TEST 1] Aspect Taxonomy Lookup:")
    taxonomy = AspectTaxonomy()
    test_terms = [
        ('battery', 'electronics'),
        ('food', 'restaurants'),
        ('staff', 'hotels'),
        ('price', 'electronics'),
        ('cleanliness', 'hotels'),
    ]
    for term, domain in test_terms:
        cat = taxonomy.get_category(term, domain)
        equiv_rest = taxonomy.get_domain_equivalents(term, domain, 'restaurants')
        equiv_hotel = taxonomy.get_domain_equivalents(term, domain, 'hotels')
        print(f"  '{term}' ({domain}) → category='{cat}'")
        if equiv_rest:
            print(f"    ↳ restaurants equiv: {equiv_rest[:2]}")
        if equiv_hotel:
            print(f"    ↳ hotels equiv:      {equiv_hotel[:2]}")

    # ── Test 2: Pivot features
    print("\n[TEST 2] Cross-Domain Pivot Features:")
    for pair in ['electronics→restaurants', 'electronics→hotels', 'restaurants→hotels']:
        src, tgt = pair.split('→')
        pivots = engine.bridge_pivots(src, tgt)
        print(f"  {pair}: {sorted(pivots)[:8]}")

    # ── Test 3: Knowledge transfer report
    print("\n[TEST 3] Transfer Knowledge Report (electronics → hotels):")
    report = engine.get_transferable_knowledge('electronics', 'hotels')
    for cat_info in report['transferable_categories'][:4]:
        print(f"  [{cat_info['category']}]")
        print(f"    electronics: {cat_info['source_terms']}")
        print(f"    hotels:      {cat_info['target_terms']}")

    # ── Test 4: Full enriched prediction
    print("\n[TEST 4] Enriched Aspect Extraction with Transfer Metadata:")
    model = RuleBasedAspectModel(weighter)
    try:
        model = RuleBasedAspectModel.load()
    except Exception:
        pass
    sentiment = SentimentPipeline(weighter)

    test_cases = [
        ("The battery life is great but the screen is dim and price is too high.", "electronics"),
        ("Staff were incredibly friendly and room was spotless but wifi was slow.", "hotels"),
        ("Food was amazing but service was terrible and ambiance felt cheap.", "restaurants"),
    ]

    for text, domain in test_cases:
        aspects = model.predict(text, domain)
        aspects = sentiment.analyze(text, domain, aspects)
        aspects = engine.enrich_aspects(aspects, domain)
        print(f"\n  [{domain}] \"{text[:60]}...\"")
        for asp in aspects:
            cat = asp.get('category', 'other')
            print(f"    '{asp['term']:18s}' "
                  f"[{asp['polarity']:8s}] "
                  f"cat={cat:12s} "
                  f"score={asp.get('score', 0):+.3f}")

    # ── Test 5: Instance transfer weights
    print("\n[TEST 5] Instance Transfer Weights (electronics → hotels):")
    weights = engine.get_instance_transfer_weights('electronics', 'hotels', splits)
    for w, rec in zip(weights[:5], splits['electronics']['train'][:5]):
        print(f"  weight={w:.3f} | \"{rec['text'][:55]}\"")

    engine.save()
    print("\n✅ Module 5 complete.")
