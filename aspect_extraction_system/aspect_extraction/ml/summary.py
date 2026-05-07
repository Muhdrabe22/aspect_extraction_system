"""
Module 6: Summary Generator
Generates human-readable summaries from extracted aspects + sentiments.

Approaches:
1. Template-based summary (fast, structured)
2. Aggregated multi-review summary (batch)
3. LLM-enhanced summary via Anthropic API (optional, high quality)
4. Aspect frequency + sentiment stats report
"""

import os
import sys
import re
import json
import pickle
from collections import defaultdict, Counter
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# ─────────────────────────────────────────────
# 1. Template Engine
# ─────────────────────────────────────────────

class TemplateSummaryEngine:
    """
    Rule-based template summary generator.
    Produces structured, readable summaries from aspect data.
    Fast, deterministic, no API needed.
    """

    OPENERS = {
        'positive': [
            "This review is largely positive.",
            "Customers are generally satisfied.",
            "The overall reception is favorable.",
            "Reviewers highlight several strengths.",
        ],
        'negative': [
            "This review raises significant concerns.",
            "Customers express notable dissatisfaction.",
            "The overall reception is unfavorable.",
            "Several issues are highlighted by reviewers.",
        ],
        'neutral': [
            "This review presents a mixed picture.",
            "Customers have both positive and negative observations.",
            "The overall reception is balanced.",
            "Reviewers note both strengths and areas for improvement.",
        ],
        'mixed': [
            "This review is mixed, with highlights and drawbacks.",
            "Customers appreciate some aspects while criticizing others.",
            "The review presents contrasting opinions across different features.",
        ]
    }

    CATEGORY_LABELS = {
        'quality':      'Quality',
        'performance':  'Performance',
        'price':        'Value for Money',
        'staff':        'Staff & Service',
        'ambiance':     'Ambiance & Atmosphere',
        'comfort':      'Comfort',
        'connectivity': 'Connectivity',
        'cleanliness':  'Cleanliness',
        'other':        'Other Aspects',
    }

    SENTIMENT_PHRASES = {
        'positive': ['praised', 'highlighted as a strength', 'well-received', 'commended'],
        'negative': ['criticized', 'flagged as a concern', 'noted as a weakness', 'raised as an issue'],
        'neutral':  ['mentioned', 'noted', 'observed', 'referenced'],
        'conflict': ['received mixed opinions on', 'drew conflicting views on', 'was controversial regarding'],
    }

    def generate(self, text, domain, aspects, overall_sentiment=None):
        """
        Generate a single-review summary.
        Returns dict with short_summary, detailed_summary, key_points, stats.
        """
        if not aspects:
            return {
                'short_summary': f"No specific aspects were identified in this {domain} review.",
                'detailed_summary': f"The review text was analyzed but no clear aspect terms could be extracted.",
                'key_points': [],
                'stats': {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0, 'conflict': 0},
                'overall_sentiment': 'neutral',
                'domain': domain,
            }

        # Compute stats
        stats = self._compute_stats(aspects)
        if overall_sentiment is None:
            overall_sentiment = self._determine_overall(stats)

        # Group aspects by sentiment
        positives = [a for a in aspects if a.get('polarity') == 'positive']
        negatives = [a for a in aspects if a.get('polarity') == 'negative']
        neutrals  = [a for a in aspects if a.get('polarity') == 'neutral']
        conflicts = [a for a in aspects if a.get('polarity') == 'conflict']

        # Build short summary
        short_summary = self._build_short(
            domain, positives, negatives, overall_sentiment, stats
        )

        # Build detailed summary
        detailed_summary = self._build_detailed(
            domain, positives, negatives, neutrals, conflicts, overall_sentiment
        )

        # Key points
        key_points = self._extract_key_points(aspects)

        return {
            'short_summary': short_summary,
            'detailed_summary': detailed_summary,
            'key_points': key_points,
            'stats': stats,
            'overall_sentiment': overall_sentiment,
            'domain': domain,
            'aspect_count': len(aspects),
        }

    def _build_short(self, domain, positives, negatives, sentiment, stats):
        """Build 1-2 sentence short summary."""
        import random
        random.seed(42)

        opener = random.choice(self.OPENERS.get(
            sentiment if sentiment in self.OPENERS else 'neutral', self.OPENERS['neutral']
        ))

        parts = []
        if positives:
            top_pos = positives[0]['term']
            if len(positives) > 1:
                parts.append(f"{top_pos} and {len(positives)-1} other aspect(s) received positive feedback")
            else:
                parts.append(f"{top_pos} received positive feedback")

        if negatives:
            top_neg = negatives[0]['term']
            if len(negatives) > 1:
                parts.append(f"while {top_neg} and {len(negatives)-1} other(s) were criticized")
            else:
                parts.append(f"while {top_neg} was criticized")

        if not parts:
            return opener + f" The review covers {stats['total']} aspect(s) of {domain} experience."

        return opener + " " + ", ".join(parts) + "."

    def _build_detailed(self, domain, positives, negatives, neutrals, conflicts, sentiment):
        """Build multi-sentence detailed summary."""
        paras = []

        # Positive aspects
        if positives:
            terms = [f"**{a['term']}**" for a in positives[:4]]
            scores = [a.get('score', 0) for a in positives[:4]]
            strongest = positives[0]['term'] if positives else ''
            avg_score = sum(scores) / len(scores) if scores else 0

            if len(positives) == 1:
                paras.append(
                    f"On the positive side, **{strongest}** was particularly appreciated "
                    f"(sentiment score: {avg_score:+.2f})."
                )
            else:
                paras.append(
                    f"On the positive side, {', '.join(terms[:-1])} and {terms[-1]} "
                    f"were all well-received, with **{strongest}** standing out most strongly."
                )

        # Negative aspects
        if negatives:
            terms = [f"**{a['term']}**" for a in negatives[:4]]
            worst = negatives[0]['term'] if negatives else ''
            worst_score = negatives[0].get('score', 0) if negatives else 0

            if len(negatives) == 1:
                paras.append(
                    f"The main concern was **{worst}** "
                    f"(sentiment score: {worst_score:+.2f})."
                )
            else:
                paras.append(
                    f"Areas of concern included {', '.join(terms[:-1])} and {terms[-1]}, "
                    f"with **{worst}** receiving the most negative feedback."
                )

        # Conflicts
        if conflicts:
            c_terms = [f"**{a['term']}**" for a in conflicts]
            paras.append(
                f"Mixed or conflicting opinions were expressed regarding {', '.join(c_terms)}."
            )

        # Neutral
        if neutrals and not (positives or negatives):
            n_terms = [a['term'] for a in neutrals[:3]]
            paras.append(
                f"The review mentions {', '.join(n_terms)} without strong opinion either way."
            )

        # Closing
        if sentiment == 'positive':
            paras.append(f"Overall, this {domain} review is positive and recommending.")
        elif sentiment == 'negative':
            paras.append(f"Overall, this {domain} review expresses dissatisfaction.")
        else:
            paras.append(f"Overall, this {domain} review is balanced with both positives and negatives.")

        return " ".join(paras)

    def _extract_key_points(self, aspects):
        """Extract top key points sorted by confidence."""
        key_points = []
        seen = set()
        sorted_aspects = sorted(
            aspects,
            key=lambda a: abs(a.get('score', 0)),
            reverse=True
        )
        for asp in sorted_aspects[:6]:
            term = asp['term']
            if term.lower() in seen:
                continue
            seen.add(term.lower())
            polarity = asp.get('polarity', 'neutral')
            score = asp.get('score', 0)
            emoji = '✅' if polarity == 'positive' else ('❌' if polarity == 'negative' else ('⚠️' if polarity == 'conflict' else '➖'))
            key_points.append({
                'term': term,
                'polarity': polarity,
                'score': round(score, 3),
                'emoji': emoji,
                'category': asp.get('category', 'other'),
                'label': f"{emoji} {term.title()} — {polarity.title()}"
            })
        return key_points

    def _compute_stats(self, aspects):
        counts = Counter(a.get('polarity', 'neutral') for a in aspects)
        return {
            'total': len(aspects),
            'positive': counts.get('positive', 0),
            'negative': counts.get('negative', 0),
            'neutral': counts.get('neutral', 0),
            'conflict': counts.get('conflict', 0),
        }

    def _determine_overall(self, stats):
        pos = stats['positive']
        neg = stats['negative']
        if pos == 0 and neg == 0:
            return 'neutral'
        if pos > neg * 1.5:
            return 'positive'
        if neg > pos * 1.5:
            return 'negative'
        return 'mixed'


# ─────────────────────────────────────────────
# 2. Batch / Multi-Review Aggregator
# ─────────────────────────────────────────────

class BatchSummaryAggregator:
    """
    Aggregates multiple review analyses into a single summary.
    Useful for CSV uploads with many reviews.
    """

    def aggregate(self, review_results, domain):
        """
        Input: list of {text, aspects: [{term, polarity, score, category}]}
        Returns: aggregated report
        """
        all_aspects = []
        for r in review_results:
            all_aspects.extend(r.get('aspects', []))

        if not all_aspects:
            return self._empty_report(domain)

        # Aggregate by term
        term_data = defaultdict(lambda: {
            'count': 0, 'scores': [], 'polarities': [], 'category': 'other'
        })
        for asp in all_aspects:
            term = asp['term'].lower().strip()
            term_data[term]['count'] += 1
            term_data[term]['scores'].append(asp.get('score', 0))
            term_data[term]['polarities'].append(asp.get('polarity', 'neutral'))
            term_data[term]['category'] = asp.get('category', 'other')

        # Build aspect report
        aspect_report = []
        for term, data in term_data.items():
            avg_score = sum(data['scores']) / len(data['scores'])
            dominant_polarity = Counter(data['polarities']).most_common(1)[0][0]
            aspect_report.append({
                'term': term,
                'mention_count': data['count'],
                'avg_score': round(avg_score, 3),
                'dominant_polarity': dominant_polarity,
                'category': data['category'],
                'sentiment_breakdown': dict(Counter(data['polarities']))
            })

        # Sort by mention count then score magnitude
        aspect_report.sort(key=lambda x: (x['mention_count'], abs(x['avg_score'])), reverse=True)

        # Overall stats
        total_reviews = len(review_results)
        all_polarities = [a.get('polarity', 'neutral') for a in all_aspects]
        polarity_counts = Counter(all_polarities)
        pos_ratio = polarity_counts.get('positive', 0) / len(all_aspects) if all_aspects else 0

        overall = 'positive' if pos_ratio > 0.6 else ('negative' if pos_ratio < 0.35 else 'mixed')

        # Top praised & criticized
        praised = [a for a in aspect_report if a['dominant_polarity'] == 'positive'][:5]
        criticized = [a for a in aspect_report if a['dominant_polarity'] == 'negative'][:5]

        # Category breakdown
        category_stats = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0})
        for asp in all_aspects:
            cat = asp.get('category', 'other')
            pol = asp.get('polarity', 'neutral')
            category_stats[cat]['total'] += 1
            if pol in ('positive', 'negative', 'neutral'):
                category_stats[cat][pol] += 1

        return {
            'domain': domain,
            'total_reviews': total_reviews,
            'total_aspects_extracted': len(all_aspects),
            'unique_aspects': len(aspect_report),
            'overall_sentiment': overall,
            'positive_ratio': round(pos_ratio, 3),
            'polarity_counts': dict(polarity_counts),
            'top_aspects': aspect_report[:10],
            'most_praised': praised,
            'most_criticized': criticized,
            'category_breakdown': {
                cat: dict(stats)
                for cat, stats in category_stats.items()
            },
            'generated_at': datetime.now().isoformat(),
        }

    def _empty_report(self, domain):
        return {
            'domain': domain,
            'total_reviews': 0,
            'total_aspects_extracted': 0,
            'unique_aspects': 0,
            'overall_sentiment': 'neutral',
            'positive_ratio': 0,
            'polarity_counts': {},
            'top_aspects': [],
            'most_praised': [],
            'most_criticized': [],
            'category_breakdown': {},
            'generated_at': datetime.now().isoformat(),
        }

    def narrative(self, agg_report):
        """Convert aggregated report to readable narrative paragraph."""
        d = agg_report
        total = d['total_reviews']
        domain = d['domain']
        overall = d['overall_sentiment']
        unique = d['unique_aspects']
        praised = [a['term'] for a in d['most_praised'][:3]]
        criticized = [a['term'] for a in d['most_criticized'][:3]]

        if total == 0:
            return "No reviews were analyzed."

        lines = []
        lines.append(
            f"Analysis of {total} {domain} review(s) identified {unique} unique aspects. "
            f"The overall sentiment is **{overall}** "
            f"with {round(d['positive_ratio'] * 100)}% positive aspect mentions."
        )

        if praised:
            lines.append(
                f"Most praised aspects: **{', '.join(praised)}**."
            )
        if criticized:
            lines.append(
                f"Most criticized aspects: **{', '.join(criticized)}**."
            )

        cats = d.get('category_breakdown', {})
        strong_cats = [
            cat for cat, stats in cats.items()
            if stats.get('positive', 0) > stats.get('negative', 0) and stats.get('total', 0) >= 2
        ]
        weak_cats = [
            cat for cat, stats in cats.items()
            if stats.get('negative', 0) > stats.get('positive', 0) and stats.get('total', 0) >= 2
        ]
        if strong_cats:
            lines.append(f"Strong categories: {', '.join(strong_cats[:3])}.")
        if weak_cats:
            lines.append(f"Categories needing improvement: {', '.join(weak_cats[:3])}.")

        return " ".join(lines)


# ─────────────────────────────────────────────
# 3. LLM Summary (Anthropic API)
# ─────────────────────────────────────────────

def generate_llm_summary(text, domain, aspects, api_key=None):
    """
    Generate high-quality summary using Claude API.
    Falls back gracefully if API unavailable.
    Returns summary string.
    """
    try:
        import urllib.request
        import json as json_mod

        aspect_lines = "\n".join(
            f"- {a['term']} [{a.get('polarity','neutral')}] score={a.get('score',0):+.2f}"
            for a in aspects[:10]
        )

        prompt = f"""You are analyzing a product/service review.

Domain: {domain}
Review text: "{text}"

Extracted aspects with sentiment:
{aspect_lines}

Write a concise 2-3 sentence summary that:
1. States the overall sentiment
2. Highlights the top positive aspect(s)
3. Mentions the main concern(s) if any
4. Is natural and readable

Summary:"""

        payload = json_mod.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}]
        }).encode('utf-8')

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json_mod.loads(resp.read().decode())
            return data['content'][0]['text'].strip()

    except Exception as e:
        return None   # Caller should fall back to template summary


# ─────────────────────────────────────────────
# 4. Summary Pipeline (Main)
# ─────────────────────────────────────────────

class SummaryPipeline:
    """
    Master summary pipeline.
    Tries LLM summary first, falls back to template.
    """

    def __init__(self, use_llm=True):
        self.template_engine = TemplateSummaryEngine()
        self.aggregator = BatchSummaryAggregator()
        self.use_llm = use_llm

    def summarize(self, text, domain, aspects, use_llm=None):
        """
        Generate complete summary for a single review.
        Returns full summary dict.
        """
        if use_llm is None:
            use_llm = self.use_llm

        # Always build template summary (fast, reliable)
        template_result = self.template_engine.generate(text, domain, aspects)

        # Try LLM enhancement
        llm_summary = None
        if use_llm and aspects:
            llm_summary = generate_llm_summary(text, domain, aspects)

        return {
            **template_result,
            'llm_summary': llm_summary,
            'final_summary': llm_summary or template_result['short_summary'],
            'has_llm': llm_summary is not None,
        }

    def summarize_batch(self, review_results, domain):
        """
        Generate aggregated summary for multiple reviews.
        """
        agg = self.aggregator.aggregate(review_results, domain)
        narrative = self.aggregator.narrative(agg)
        return {**agg, 'narrative': narrative}

    def save(self, path=None):
        if path is None:
            path = os.path.join(Config.MODEL_DIR, 'summary_pipeline.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[SUMMARY] Saved to {path}")

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = os.path.join(Config.MODEL_DIR, 'summary_pipeline.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("Module 6: Summary Generator")
    print("=" * 55)

    from ml.feature_weighting import FeatureWeightComputer
    from ml.model import RuleBasedAspectModel
    from ml.sentiment import SentimentPipeline
    from ml.knowledge_transfer import KnowledgeTransferEngine
    from ml.dataset import load_processed, build_dataset

    try:
        dataset, splits = load_processed()
    except Exception:
        dataset, _ = build_dataset()

    try:
        weighter = FeatureWeightComputer.load()
    except Exception:
        weighter = FeatureWeightComputer().fit(dataset)
        weighter.save()

    try:
        transfer_engine = KnowledgeTransferEngine.load()
    except Exception:
        transfer_engine = KnowledgeTransferEngine(weighter)
        transfer_engine.fit(dataset)
        transfer_engine.save()

    model = RuleBasedAspectModel(weighter)
    try:
        model = RuleBasedAspectModel.load()
    except Exception:
        pass

    sentiment_pipeline = SentimentPipeline(weighter)
    summary_pipeline = SummaryPipeline(use_llm=False)

    # ── Test 1: Single review summaries
    print("\n[TEST 1] Single Review Summaries:\n")
    test_cases = [
        ("The battery life is amazing and the screen is beautiful, but the price feels too high for what you get.", "electronics"),
        ("Service was incredibly slow and staff were rude. However, the food itself was absolutely delicious.", "restaurants"),
        ("Spotless room with a stunning view. Breakfast was excellent. Wifi was terrible and parking was a nightmare.", "hotels"),
        ("Camera takes stunning photos and performance is blazing fast. Build quality feels cheap though.", "electronics"),
    ]

    for text, domain in test_cases:
        aspects = model.predict(text, domain)
        aspects = sentiment_pipeline.analyze(text, domain, aspects)
        aspects = transfer_engine.enrich_aspects(aspects, domain)
        summary = summary_pipeline.summarize(text, domain, aspects, use_llm=False)

        print(f"[{domain.upper()}]")
        print(f"  Text    : \"{text[:70]}...\"")
        print(f"  Short   : {summary['short_summary']}")
        print(f"  Detailed: {summary['detailed_summary'][:120]}...")
        print(f"  Overall : {summary['overall_sentiment'].upper()}")
        print(f"  Stats   : +{summary['stats']['positive']} / -{summary['stats']['negative']} / ~{summary['stats']['neutral']}")
        print(f"  Key pts :")
        for kp in summary['key_points']:
            print(f"    {kp['label']}  (score={kp['score']:+.3f})")
        print()

    # ── Test 2: Batch summary
    print("[TEST 2] Batch / Multi-Review Aggregated Summary:\n")
    batch_reviews = [
        "The food was excellent and service was fast. Prices are reasonable.",
        "Terrible service, very rude staff. Food was okay but cold.",
        "Amazing ambiance and great food. A bit pricey but worth it.",
        "Staff were friendly but wait time was too long. Food quality was good.",
        "Best meal I've had in years. Cleanliness was impeccable.",
    ]

    batch_results = []
    for rev in batch_reviews:
        aspects = model.predict(rev, 'restaurants')
        aspects = sentiment_pipeline.analyze(rev, 'restaurants', aspects)
        aspects = transfer_engine.enrich_aspects(aspects, 'restaurants')
        batch_results.append({'text': rev, 'aspects': aspects})

    batch_summary = summary_pipeline.summarize_batch(batch_results, 'restaurants')
    print(f"  Narrative : {batch_summary['narrative']}")
    print(f"  Overall   : {batch_summary['overall_sentiment'].upper()} "
          f"({round(batch_summary['positive_ratio']*100)}% positive)")
    print(f"  Praised   : {[a['term'] for a in batch_summary['most_praised'][:3]]}")
    print(f"  Criticized: {[a['term'] for a in batch_summary['most_criticized'][:3]]}")
    print(f"  Categories:")
    for cat, stats in list(batch_summary['category_breakdown'].items())[:4]:
        print(f"    {cat:15s}: +{stats.get('positive',0)} / -{stats.get('negative',0)}")

    summary_pipeline.save()
    print("\n✅ Module 6 complete.")
