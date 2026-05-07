"""
Module 4: Sentiment Classifier
Per-aspect sentiment classification: positive / negative / neutral / conflict

Architecture:
  Aspect term + surrounding context window
      ↓
  Feature-weighted context encoding
      ↓
  Sentiment scoring (rule-based + ML hybrid)
      ↓
  Label: positive | negative | neutral | conflict

Works standalone or as post-processing step on Module 3 output.
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
from ml.dataset import SENTIMENT_LABEL2ID, SENTIMENT_ID2LABEL

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizerFast, BertModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────
# 1. Lexicon-Based Sentiment Engine
# ─────────────────────────────────────────────

class SentimentLexicon:
    """
    Rich opinion lexicon with:
    - Polarity scores (-1 to +1)
    - Intensifiers & diminishers
    - Negation handling
    - Domain-specific overrides
    """

    def __init__(self):
        self.positive = {
            # Strong positive
            'excellent': 0.95, 'outstanding': 0.95, 'exceptional': 0.95,
            'perfect': 0.90, 'superb': 0.90, 'brilliant': 0.90,
            'amazing': 0.88, 'fantastic': 0.88, 'wonderful': 0.88,
            'incredible': 0.85, 'stunning': 0.85, 'impressive': 0.85,
            # Moderate positive
            'great': 0.80, 'good': 0.75, 'nice': 0.70, 'lovely': 0.72,
            'comfortable': 0.70, 'clean': 0.68, 'fast': 0.65,
            'helpful': 0.72, 'friendly': 0.72, 'fresh': 0.65,
            'delicious': 0.85, 'tasty': 0.78, 'spacious': 0.70,
            'convenient': 0.68, 'smooth': 0.65, 'prompt': 0.70,
            'reasonable': 0.62, 'generous': 0.72, 'vivid': 0.65,
            'sharp': 0.65, 'clear': 0.62, 'reliable': 0.70,
            'efficient': 0.68, 'attentive': 0.72, 'cozy': 0.70,
            'stylish': 0.65, 'modern': 0.60, 'heavenly': 0.88,
            'love': 0.82, 'loved': 0.82, 'enjoy': 0.72, 'enjoyed': 0.72,
            'recommend': 0.75, 'recommended': 0.75, 'worth': 0.65,
            'satisfied': 0.70, 'happy': 0.75, 'pleased': 0.72,
        }

        self.negative = {
            # Strong negative
            'terrible': -0.95, 'horrible': -0.95, 'awful': -0.95,
            'dreadful': -0.92, 'atrocious': -0.92, 'appalling': -0.90,
            'disgusting': -0.90, 'unacceptable': -0.88,
            # Moderate negative
            'bad': -0.80, 'poor': -0.78, 'worst': -0.90,
            'disappointing': -0.75, 'disappointed': -0.75,
            'slow': -0.65, 'dirty': -0.80, 'rude': -0.85,
            'broken': -0.80, 'noisy': -0.65, 'expensive': -0.60,
            'overpriced': -0.75, 'outdated': -0.65, 'old': -0.45,
            'cold': -0.60, 'bland': -0.72, 'stale': -0.72,
            'unhelpful': -0.80, 'unfriendly': -0.78, 'rotten': -0.88,
            'smelled': -0.75, 'smells': -0.75, 'crowded': -0.55,
            'uncomfortable': -0.70, 'cramped': -0.65, 'late': -0.60,
            'dim': -0.60, 'weak': -0.60, 'cheap': -0.55,
            'nightmare': -0.88, 'waste': -0.80, 'avoid': -0.82,
            'never': -0.70, 'problem': -0.65, 'issue': -0.60,
            'complain': -0.72, 'complaint': -0.72, 'failed': -0.80,
        }

        self.intensifiers = {
            'very': 1.3, 'extremely': 1.5, 'incredibly': 1.4,
            'absolutely': 1.4, 'totally': 1.3, 'completely': 1.3,
            'quite': 1.15, 'rather': 1.1, 'pretty': 1.1,
            'so': 1.2, 'too': 1.25, 'highly': 1.3,
            'super': 1.35, 'really': 1.25, 'truly': 1.3,
            'exceptionally': 1.5, 'remarkably': 1.4,
        }

        self.diminishers = {
            'slightly': 0.6, 'somewhat': 0.65, 'a bit': 0.65,
            'a little': 0.65, 'barely': 0.4, 'hardly': 0.4,
            'almost': 0.75, 'nearly': 0.75, 'kind of': 0.7,
            'sort of': 0.7, 'not very': 0.5, 'not too': 0.5,
        }

        self.negations = {
            'not', 'no', 'never', 'nothing', 'nobody', 'neither',
            "n't", "wasn't", "isn't", "didn't", "don't", "won't",
            "wouldn't", "couldn't", "shouldn't", "doesn't", "aren't",
            "weren't", "haven't", "hadn't", "can't", "cannot",
        }

        # Domain-specific overrides
        self.domain_overrides = {
            'electronics': {
                'fast': 0.80,    # faster is better for electronics
                'slow': -0.80,
                'dim': -0.70,
                'bright': 0.72,
                'loud': 0.65,    # loud speaker is good
                'quiet': -0.40,  # quiet speaker is bad
            },
            'restaurants': {
                'loud': -0.60,   # loud restaurant is bad
                'quiet': 0.60,   # quiet restaurant is good
                'hot': 0.55,     # hot food is good
                'cold': -0.70,   # cold food is bad
                'fresh': 0.85,
                'greasy': -0.65,
            },
            'hotels': {
                'quiet': 0.65,
                'noisy': -0.80,
                'clean': 0.80,
                'dirty': -0.90,
                'spacious': 0.75,
                'small': -0.55,
            }
        }

    def score(self, word, domain=None):
        """Return polarity score for a word (-1 to +1)."""
        w = word.lower().rstrip('.,!?;:')

        # Domain override first
        if domain and domain in self.domain_overrides:
            if w in self.domain_overrides[domain]:
                return self.domain_overrides[domain][w]

        if w in self.positive:
            return self.positive[w]
        if w in self.negative:
            return self.negative[w]
        return 0.0

    def intensifier(self, word):
        return self.intensifiers.get(word.lower(), 1.0)

    def is_negation(self, word):
        return word.lower().rstrip("'") in self.negations


# ─────────────────────────────────────────────
# 2. Context Window Extractor
# ─────────────────────────────────────────────

class ContextWindowExtractor:
    """
    Extracts relevant context around an aspect term for sentiment analysis.
    Handles sentence boundaries and clause detection.
    """

    def __init__(self, window=6):
        self.window = window

    def extract(self, tokens, aspect_start, aspect_end):
        """
        Extract context window around aspect span [aspect_start, aspect_end].
        Returns (left_context, aspect_tokens, right_context)
        """
        left_start = max(0, aspect_start - self.window)
        right_end = min(len(tokens), aspect_end + self.window + 1)

        left_ctx = tokens[left_start:aspect_start]
        aspect_toks = tokens[aspect_start:aspect_end + 1]
        right_ctx = tokens[aspect_end + 1:right_end]

        # Truncate at clause boundary (but, however, although, etc.)
        clause_markers = {'but', 'however', 'although', 'though', 'yet', 'while', 'whereas'}
        left_ctx = self._truncate_at_clause(left_ctx, clause_markers, from_right=True)
        right_ctx = self._truncate_at_clause(right_ctx, clause_markers, from_right=False)

        return left_ctx, aspect_toks, right_ctx

    def _truncate_at_clause(self, tokens, markers, from_right=False):
        if from_right:
            for i in range(len(tokens) - 1, -1, -1):
                if tokens[i].lower() in markers:
                    return tokens[i + 1:]
        else:
            for i, t in enumerate(tokens):
                if t.lower() in markers:
                    return tokens[:i]
        return tokens


# ─────────────────────────────────────────────
# 3. Core Sentiment Scorer
# ─────────────────────────────────────────────

class SentimentScorer:
    """
    Computes sentiment score for an aspect given its context.
    Uses lexicon + intensifiers + negation + feature weights.
    """

    def __init__(self, lexicon: SentimentLexicon = None):
        self.lexicon = lexicon or SentimentLexicon()
        self.extractor = ContextWindowExtractor()

    def score_context(self, tokens, domain=None, weights=None):
        """
        Score sentiment of a token sequence.
        Returns float score (-1 to +1)
        """
        if not tokens:
            return 0.0

        if weights is None:
            weights = [1.0] * len(tokens)

        total_score = 0.0
        total_weight = 0.0
        negated = False
        intensifier = 1.0

        for i, token in enumerate(tokens):
            t_lower = token.lower().rstrip('.,!?;:')
            w = weights[i] if i < len(weights) else 1.0

            # Check negation
            if self.lexicon.is_negation(token):
                negated = True
                continue

            # Check intensifier
            intens = self.lexicon.intensifier(token)
            if intens != 1.0:
                intensifier = intens
                continue

            # Get polarity score
            score = self.lexicon.score(token, domain)
            if score != 0.0:
                # Apply negation
                if negated:
                    score = -score * 0.8
                    negated = False
                # Apply intensifier
                score *= intensifier
                intensifier = 1.0
                # Apply feature weight bonus
                effective_weight = 1.0 + w * 0.5
                total_score += score * effective_weight
                total_weight += effective_weight
            else:
                negated = False
                intensifier = 1.0

        if total_weight == 0:
            return 0.0
        return total_score / total_weight

    def classify(self, score, threshold=0.15):
        """Convert numeric score to sentiment label."""
        if score > threshold:
            return 'positive'
        elif score < -threshold:
            return 'negative'
        return 'neutral'

    def score_aspect(self, tokens, aspect_start, aspect_end,
                     domain=None, weights=None):
        """
        Full aspect sentiment scoring with context extraction.
        Returns dict with score, label, confidence, context.
        """
        left_ctx, aspect_toks, right_ctx = self.extractor.extract(
            tokens, aspect_start, aspect_end
        )

        # Weighted context: right context is more relevant
        left_score = self.score_context(left_ctx, domain, weights)
        right_score = self.score_context(right_ctx, domain, weights)

        # Right context weighted higher (aspect → opinion)
        combined = left_score * 0.35 + right_score * 0.65

        label = self.classify(combined)
        confidence = min(abs(combined) * 2, 1.0)

        return {
            'score': round(combined, 4),
            'polarity': label,
            'confidence': round(confidence, 3),
            'left_context': ' '.join(left_ctx),
            'right_context': ' '.join(right_ctx),
        }


# ─────────────────────────────────────────────
# 4. Multi-Aspect Conflict Detector
# ─────────────────────────────────────────────

class ConflictDetector:
    """
    Detects conflicting sentiment for the same aspect.
    e.g. "The food is great but the portions are small"
    → food=positive, portions=negative (no conflict)
    e.g. "The room is nice but also quite noisy"
    → room=conflict (same aspect, mixed signals)
    """

    def detect(self, aspects_with_sentiment):
        """
        Input: list of {term, polarity, score}
        Output: same list with conflicts flagged
        """
        # Group by normalized term
        term_scores = defaultdict(list)
        for asp in aspects_with_sentiment:
            key = asp['term'].lower().rstrip('.,!?')
            term_scores[key].append(asp)

        result = []
        for asp in aspects_with_sentiment:
            key = asp['term'].lower().rstrip('.,!?')
            group = term_scores[key]

            if len(group) > 1:
                polarities = set(a['polarity'] for a in group)
                if 'positive' in polarities and 'negative' in polarities:
                    asp = {**asp, 'polarity': 'conflict'}

            result.append(asp)
        return result


# ─────────────────────────────────────────────
# 5. Full Sentiment Pipeline
# ─────────────────────────────────────────────

class SentimentPipeline:
    """
    End-to-end sentiment analysis for extracted aspects.
    Input: text + domain + list of extracted aspects (from Module 3)
    Output: aspects enriched with sentiment scores
    """

    def __init__(self, weighter=None):
        self.lexicon = SentimentLexicon()
        self.scorer = SentimentScorer(self.lexicon)
        self.conflict_detector = ConflictDetector()
        self.weighter = weighter

    def analyze(self, text, domain, aspects):
        """
        Analyze sentiment for each aspect in the text.

        aspects: list of {term, start_token, end_token, ...}
        Returns: enriched aspects with polarity + score + confidence
        """
        tokens = re.findall(r'\S+', text)

        # Get feature weights
        weights = [1.0] * len(tokens)
        if self.weighter:
            try:
                weights = self.weighter.compute_token_weights(tokens, domain)
            except Exception:
                pass

        enriched = []
        for asp in aspects:
            start = asp.get('start_token', 0)
            end = asp.get('end_token', start)

            # Score sentiment using context
            result = self.scorer.score_aspect(
                tokens, start, end, domain, weights
            )

            enriched.append({
                **asp,
                'polarity': result['polarity'],
                'score': result['score'],
                'confidence': result['confidence'],
                'context': {
                    'left': result['left_context'],
                    'right': result['right_context'],
                }
            })

        # Detect conflicts
        enriched = self.conflict_detector.detect(enriched)
        return enriched

    def analyze_raw(self, text, domain):
        """
        Full pipeline: extract aspects + analyze sentiment.
        Uses Module 3 model for extraction.
        """
        from ml.model import RuleBasedAspectModel

        try:
            weighter = self.weighter
            model = RuleBasedAspectModel(weighter)
            try:
                model = RuleBasedAspectModel.load()
            except Exception:
                pass
            aspects = model.predict(text, domain)
        except Exception as e:
            print(f"[WARN] Extraction failed: {e}")
            aspects = []

        return self.analyze(text, domain, aspects)

    def batch_analyze(self, reviews, domain):
        """
        Analyze a list of review texts.
        Returns list of {text, aspects_with_sentiment}
        """
        results = []
        for review in reviews:
            aspects = self.analyze_raw(review, domain)
            results.append({
                'text': review,
                'domain': domain,
                'aspects': aspects,
                'aspect_count': len(aspects),
                'sentiment_summary': self._summarize(aspects)
            })
        return results

    def _summarize(self, aspects):
        """Aggregate sentiment counts across aspects."""
        counts = defaultdict(int)
        for asp in aspects:
            counts[asp['polarity']] += 1
        total = sum(counts.values())
        if total == 0:
            return {'positive': 0, 'negative': 0, 'neutral': 0, 'conflict': 0, 'total': 0}
        return {
            'positive': counts['positive'],
            'negative': counts['negative'],
            'neutral': counts['neutral'],
            'conflict': counts['conflict'],
            'total': total,
            'overall': self._overall_sentiment(aspects)
        }

    def _overall_sentiment(self, aspects):
        """Compute overall review sentiment from aspect scores."""
        if not aspects:
            return 'neutral'
        scores = [a.get('score', 0) for a in aspects]
        avg = sum(scores) / len(scores)
        if avg > 0.1:
            return 'positive'
        elif avg < -0.1:
            return 'negative'
        return 'neutral'

    def save(self, path=None):
        if path is None:
            path = os.path.join(Config.MODEL_DIR, 'sentiment_pipeline.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[SENTIMENT] Saved to {path}")

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = os.path.join(Config.MODEL_DIR, 'sentiment_pipeline.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)


# ─────────────────────────────────────────────
# 6. Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("Module 4: Sentiment Classifier")
    print("=" * 55)

    from ml.feature_weighting import FeatureWeightComputer
    try:
        weighter = FeatureWeightComputer.load()
    except Exception:
        from ml.dataset import load_processed, build_dataset
        try:
            dataset, _ = load_processed()
        except Exception:
            dataset, _ = build_dataset()
        weighter = FeatureWeightComputer().fit(dataset)
        weighter.save()

    pipeline = SentimentPipeline(weighter)

    test_cases = [
        ("The battery life is amazing but the screen is too dim.", "electronics"),
        ("Service was extremely slow but the food tasted absolutely incredible.", "restaurants"),
        ("The room was very spacious. However, the wifi was terrible and the staff were quite rude.", "hotels"),
        ("Camera quality is stunning and the price is very reasonable.", "electronics"),
        ("Breakfast was excellent. The parking was a complete nightmare though.", "hotels"),
        ("The keyboard is comfortable but the battery drains incredibly fast.", "electronics"),
    ]

    print("\n[TEST] Full Aspect + Sentiment Analysis:\n")
    for text, domain in test_cases:
        results = pipeline.analyze_raw(text, domain)
        print(f"[{domain.upper()}]")
        print(f"  Text: \"{text}\"")
        for asp in results:
            bar = '▓' * int(abs(asp['score']) * 10) if asp['score'] != 0 else '░'
            sign = '+' if asp['score'] > 0 else ''
            print(f"  → '{asp['term']:20s}' "
                  f"[{asp['polarity']:8s}] "
                  f"score={sign}{asp['score']:+.3f} "
                  f"conf={asp['confidence']:.2f}  {bar}")
        print()

    # Batch test
    print("[BATCH TEST] Restaurant Reviews:")
    reviews = [
        "Amazing food and great atmosphere. Service was a bit slow though.",
        "The worst meal I've ever had. Rude staff and cold food.",
        "Pretty good overall. Portions were generous and price was fair.",
    ]
    batch_results = pipeline.batch_analyze(reviews, 'restaurants')
    for r in batch_results:
        s = r['sentiment_summary']
        print(f"  \"{r['text'][:55]}...\"")
        print(f"  → Overall: {s['overall']} | "
              f"+{s['positive']} / -{s['negative']} / ~{s['neutral']}")
    print()

    pipeline.save()
    print("✅ Module 4 complete.")
