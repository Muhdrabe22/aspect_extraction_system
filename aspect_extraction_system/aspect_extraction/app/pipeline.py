"""
ML Pipeline Loader
Singleton loader — initializes all ML components once at startup.
Used by Flask routes to access the full pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

_pipeline = None


def get_pipeline():
    """Return singleton ML pipeline, initializing if needed."""
    global _pipeline
    if _pipeline is None:
        _pipeline = _init_pipeline()
    return _pipeline


def _init_pipeline():
    from ml.dataset import load_processed, build_dataset, split_dataset, save_processed
    from ml.feature_weighting import FeatureWeightComputer
    from ml.model import RuleBasedAspectModel
    from ml.sentiment import SentimentPipeline
    from ml.knowledge_transfer import KnowledgeTransferEngine
    from ml.summary import SummaryPipeline

    print("[PIPELINE] Initializing ML pipeline...")

    # 1. Dataset
    try:
        dataset, splits = load_processed()
    except FileNotFoundError:
        print("[PIPELINE] Building dataset...")
        dataset, _ = build_dataset()
        splits = split_dataset(dataset)
        save_processed(dataset, splits)

    # 2. Feature weighter
    try:
        weighter = FeatureWeightComputer.load()
    except FileNotFoundError:
        print("[PIPELINE] Fitting feature weighter...")
        weighter = FeatureWeightComputer().fit(dataset)
        weighter.save()

    # 3. Aspect model
    try:
        aspect_model = RuleBasedAspectModel.load()
    except Exception:
        print("[PIPELINE] Building rule-based aspect model...")
        aspect_model = RuleBasedAspectModel(weighter)
        aspect_model.save()

    # 4. Sentiment pipeline
    sentiment = SentimentPipeline(weighter)

    # 5. Knowledge transfer engine
    try:
        transfer = KnowledgeTransferEngine.load()
    except Exception:
        print("[PIPELINE] Fitting knowledge transfer engine...")
        transfer = KnowledgeTransferEngine(weighter)
        transfer.fit(dataset)
        transfer.save()

    # 6. Summary pipeline
    summary = SummaryPipeline(use_llm=True)

    print("[PIPELINE] ✅ All components ready")

    return {
        'weighter':      weighter,
        'aspect_model':  aspect_model,
        'sentiment':     sentiment,
        'transfer':      transfer,
        'summary':       summary,
    }


def analyze_review(text, domain, use_llm=True):
    """
    Full pipeline: text + domain → complete analysis dict.
    """
    import re
    pipeline = get_pipeline()

    # Clean text
    text = text.strip()
    if not text:
        return {'error': 'Empty review text'}

    # Step 1: Extract aspects
    aspects = pipeline['aspect_model'].predict(text, domain)

    # Step 2: Sentiment per aspect
    aspects = pipeline['sentiment'].analyze(text, domain, aspects)

    # Step 3: Knowledge transfer enrichment
    aspects = pipeline['transfer'].enrich_aspects(aspects, domain)

    # Step 4: Clean up aspect terms (remove leading conjunctions/articles)
    aspects = _clean_aspect_terms(aspects)

    # Step 5: Summary
    summary = pipeline['summary'].summarize(text, domain, aspects, use_llm=use_llm)

    return {
        'text':             text,
        'domain':           domain,
        'aspects':          aspects,
        'summary':          summary,
        'overall_sentiment': summary['overall_sentiment'],
        'stats':            summary['stats'],
    }


def _clean_aspect_terms(aspects):
    """Remove leading stopwords/conjunctions from extracted aspect terms."""
    import re
    stopwords = {'the', 'a', 'an', 'and', 'but', 'or', 'so', 'yet',
                 'for', 'nor', 'this', 'that', 'its', 'it', 'is', 'was'}
    cleaned = []
    for asp in aspects:
        term = asp['term']
        # Strip leading stopwords
        words = term.split()
        while words and words[0].lower().rstrip('.,!?') in stopwords:
            words = words[1:]
        # Strip trailing punctuation
        if words:
            words[-1] = words[-1].rstrip('.,!?;:')
        cleaned_term = ' '.join(words).strip()
        if cleaned_term and len(cleaned_term) > 1:
            cleaned.append({**asp, 'term': cleaned_term})
    return cleaned
