"""
Module 3: Aspect Extraction Model
BERT-based token classifier for BIO aspect tagging.
Integrates domain-aware feature weights as attention bias.
Supports multi-domain training with domain embeddings.

Architecture:
  BERT Encoder
      ↓
  Domain Embedding (injected)
      ↓
  Feature Weight Attention Bias
      ↓
  BiLSTM Contextualizer
      ↓
  CRF / Linear Classifier
      ↓
  BIO Labels  +  Sentiment Labels
"""

import os
import sys
import json
import pickle
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from ml.dataset import (
    BIO_LABEL2ID, BIO_ID2LABEL,
    SENTIMENT_LABEL2ID, SENTIMENT_ID2LABEL,
    DOMAIN_LABEL2ID, DOMAIN_ID2LABEL,
    load_processed, build_dataset, split_dataset, save_processed
)
from ml.feature_weighting import FeatureWeightComputer

# ─────────────────────────────────────────────
# Try importing torch/transformers; fall back to
# a lightweight rule-based model if not available
# ─────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizerFast, BertModel
    TORCH_AVAILABLE = True
    print("[MODEL] PyTorch + Transformers available ✅")
except ImportError:
    TORCH_AVAILABLE = False
    print("[MODEL] PyTorch not available — using rule-based fallback model")


# ═══════════════════════════════════════════════
# SECTION A: PyTorch BERT Model (full version)
# ═══════════════════════════════════════════════

if TORCH_AVAILABLE:

    class AspectDataset(Dataset):
        """
        PyTorch Dataset for aspect extraction.
        Tokenizes text with BERT tokenizer, aligns BIO labels
        to BERT subword tokens, injects feature weights.
        """

        def __init__(self, records, tokenizer, max_len=128, weighter=None):
            self.records = records
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.weighter = weighter

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            record = self.records[idx]
            tokens = record['tokens']
            bio_labels = record['bio_labels']
            sentiment_labels = record['sentiment_labels']
            domain = record['domain']
            token_weights = record.get('token_weights', [1.0] * len(tokens))

            # Encode with BERT fast tokenizer (word_ids for alignment)
            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            word_ids = encoding.word_ids()

            # Align BIO labels to subword tokens
            bio_label_ids = []
            weight_ids = []
            prev_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    bio_label_ids.append(-100)   # special token, ignore in loss
                    weight_ids.append(0.0)
                elif word_id != prev_word_id:
                    # First subword of a word → real label
                    label = bio_labels[word_id] if word_id < len(bio_labels) else 'O'
                    bio_label_ids.append(BIO_LABEL2ID.get(label, 0))
                    w = token_weights[word_id] if word_id < len(token_weights) else 1.0
                    weight_ids.append(w)
                else:
                    # Continuation subword → I-ASP if previous was B/I, else O
                    prev_label = bio_label_ids[-1]
                    if prev_label in (BIO_LABEL2ID['B-ASP'], BIO_LABEL2ID['I-ASP']):
                        bio_label_ids.append(BIO_LABEL2ID['I-ASP'])
                    else:
                        bio_label_ids.append(-100)
                    weight_ids.append(weight_ids[-1])
                prev_word_id = word_id

            # Pad to max_len
            bio_label_ids = (bio_label_ids + [-100] * self.max_len)[:self.max_len]
            weight_ids = (weight_ids + [0.0] * self.max_len)[:self.max_len]

            domain_id = DOMAIN_LABEL2ID.get(domain, 0)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'bio_labels': torch.tensor(bio_label_ids, dtype=torch.long),
                'token_weights': torch.tensor(weight_ids, dtype=torch.float),
                'domain_id': torch.tensor(domain_id, dtype=torch.long)
            }


    class AspectExtractionModel(nn.Module):
        """
        BERT + Domain Embedding + Feature Weight Bias + BiLSTM + CRF-like classifier.

        Forward pass returns:
          - bio_logits: (batch, seq_len, 3)  for BIO classification
        """

        def __init__(self, bert_model_name=None, num_domains=3, num_bio_labels=3,
                     lstm_hidden=128, dropout=0.1):
            super().__init__()
            if bert_model_name is None:
                bert_model_name = Config.BERT_MODEL_NAME

            self.bert = BertModel.from_pretrained(bert_model_name)
            bert_hidden = self.bert.config.hidden_size   # 768

            # Domain embedding: maps domain_id → 64-dim vector
            self.domain_embedding = nn.Embedding(num_domains, 64)

            # Feature weight projection: scalar weight → 64-dim vector
            self.weight_proj = nn.Linear(1, 64)

            # Fusion: bert_hidden + domain_emb + weight_emb → fused
            fused_dim = bert_hidden + 64 + 64
            self.fusion = nn.Linear(fused_dim, bert_hidden)
            self.fusion_norm = nn.LayerNorm(bert_hidden)

            # BiLSTM contextualizer
            self.bilstm = nn.LSTM(
                bert_hidden, lstm_hidden,
                num_layers=2, batch_first=True,
                bidirectional=True, dropout=dropout
            )

            # BIO classifier head
            self.dropout = nn.Dropout(dropout)
            self.bio_classifier = nn.Linear(lstm_hidden * 2, num_bio_labels)

        def forward(self, input_ids, attention_mask, domain_ids, token_weights):
            # BERT encoding
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            seq_out = bert_out.last_hidden_state   # (B, L, 768)

            B, L, H = seq_out.shape

            # Domain embedding: (B, 64) → expand to (B, L, 64)
            d_emb = self.domain_embedding(domain_ids)          # (B, 64)
            d_emb = d_emb.unsqueeze(1).expand(B, L, -1)        # (B, L, 64)

            # Feature weight embedding: (B, L, 1) → (B, L, 64)
            w = token_weights.unsqueeze(-1)                     # (B, L, 1)
            w_emb = torch.relu(self.weight_proj(w))             # (B, L, 64)

            # Fuse BERT + domain + weights
            fused = torch.cat([seq_out, d_emb, w_emb], dim=-1)  # (B, L, 896)
            fused = torch.relu(self.fusion(fused))               # (B, L, 768)
            fused = self.fusion_norm(fused)

            # BiLSTM
            lstm_out, _ = self.bilstm(fused)                    # (B, L, 256)

            # BIO logits
            out = self.dropout(lstm_out)
            bio_logits = self.bio_classifier(out)               # (B, L, 3)

            return bio_logits


    def train_model(splits, weighter, device=None, epochs=None, batch_size=None, lr=None):
        """
        Fine-tune BERT aspect extraction model on multi-domain data.
        Returns trained model + tokenizer.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if epochs is None:
            epochs = Config.NUM_EPOCHS
        if batch_size is None:
            batch_size = Config.BATCH_SIZE
        if lr is None:
            lr = Config.LEARNING_RATE

        print(f"[TRAIN] Device: {device}")

        # Load tokenizer
        print(f"[TRAIN] Loading tokenizer: {Config.BERT_MODEL_NAME}")
        tokenizer = BertTokenizerFast.from_pretrained(Config.BERT_MODEL_NAME)

        # Build combined train / val sets across all domains
        all_train, all_val = [], []
        for domain, split in splits.items():
            all_train.extend(split['train'])
            all_val.extend(split['val'])

        print(f"[TRAIN] Train: {len(all_train)} | Val: {len(all_val)}")

        train_ds = AspectDataset(all_train, tokenizer, weighter=weighter)
        val_ds = AspectDataset(all_val, tokenizer, weighter=weighter)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Model
        model = AspectExtractionModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

        for epoch in range(1, epochs + 1):
            # ── Train ──
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                domain_ids = batch['domain_id'].to(device)
                token_weights = batch['token_weights'].to(device)
                bio_labels = batch['bio_labels'].to(device)

                logits = model(input_ids, attention_mask, domain_ids, token_weights)

                loss = criterion(logits.view(-1, 3), bio_labels.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            avg_train = train_loss / len(train_loader)

            # ── Validate ──
            model.eval()
            val_loss = 0.0
            all_preds, all_true = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    domain_ids = batch['domain_id'].to(device)
                    token_weights = batch['token_weights'].to(device)
                    bio_labels = batch['bio_labels'].to(device)

                    logits = model(input_ids, attention_mask, domain_ids, token_weights)
                    loss = criterion(logits.view(-1, 3), bio_labels.view(-1))
                    val_loss += loss.item()

                    preds = torch.argmax(logits, dim=-1)
                    mask = bio_labels != -100
                    all_preds.extend(preds[mask].cpu().tolist())
                    all_true.extend(bio_labels[mask].cpu().tolist())

            avg_val = val_loss / max(len(val_loader), 1)
            f1 = _bio_f1(all_true, all_preds)

            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
            history['val_f1'].append(f1)

            print(f"[TRAIN] Epoch {epoch}/{epochs} | "
                  f"Train Loss: {avg_train:.4f} | "
                  f"Val Loss: {avg_val:.4f} | "
                  f"Val F1: {f1:.4f}")

            # Save best
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                _save_model(model, tokenizer, history)
                print(f"[TRAIN] ✅ Best model saved (val_loss={avg_val:.4f})")

        return model, tokenizer, history


    def _bio_f1(true_ids, pred_ids):
        """Compute F1 for B-ASP label."""
        tp = sum(1 for t, p in zip(true_ids, pred_ids) if t == BIO_LABEL2ID['B-ASP'] and p == BIO_LABEL2ID['B-ASP'])
        fp = sum(1 for t, p in zip(true_ids, pred_ids) if t != BIO_LABEL2ID['B-ASP'] and p == BIO_LABEL2ID['B-ASP'])
        fn = sum(1 for t, p in zip(true_ids, pred_ids) if t == BIO_LABEL2ID['B-ASP'] and p != BIO_LABEL2ID['B-ASP'])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


    def _save_model(model, tokenizer, history):
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(Config.MODEL_DIR, 'aspect_model.pt'))
        tokenizer.save_pretrained(os.path.join(Config.MODEL_DIR, 'tokenizer'))
        with open(os.path.join(Config.MODEL_DIR, 'train_history.json'), 'w') as f:
            json.dump(history, f, indent=2)


    def load_model(device=None):
        """Load saved BERT model + tokenizer."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.join(Config.MODEL_DIR, 'aspect_model.pt')
        tok_path = os.path.join(Config.MODEL_DIR, 'tokenizer')
        if not os.path.exists(model_path):
            raise FileNotFoundError("No trained model found. Run train_model() first.")
        model = AspectExtractionModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        tokenizer = BertTokenizerFast.from_pretrained(tok_path)
        return model, tokenizer


# ═══════════════════════════════════════════════
# SECTION B: Rule-Based Fallback Model
# (used when PyTorch/BERT not available)
# ═══════════════════════════════════════════════

class RuleBasedAspectModel:
    """
    Lightweight rule-based aspect extractor.
    Uses feature weights + POS-like heuristics + seed matching.
    No GPU / BERT required. Suitable for demo + production fallback.

    Accuracy is lower than BERT but still useful for:
    - Demo without GPU
    - Fast inference
    - Explainable predictions
    """

    def __init__(self, weighter: FeatureWeightComputer = None):
        self.weighter = weighter
        self._load_pos_patterns()

    def _load_pos_patterns(self):
        """Common aspect-indicating patterns."""
        import re
        # Nouns that are often aspects
        self.aspect_indicators = {
            'electronics': {
                'battery', 'screen', 'camera', 'performance', 'price',
                'design', 'software', 'storage', 'display', 'keyboard',
                'speaker', 'charging', 'processor', 'memory', 'build',
                'quality', 'sound', 'display', 'button', 'port', 'wifi',
                'bluetooth', 'weight', 'size', 'color', 'material'
            },
            'restaurants': {
                'food', 'service', 'ambiance', 'price', 'menu',
                'staff', 'location', 'cleanliness', 'portion', 'taste',
                'drinks', 'dessert', 'wait', 'reservation', 'atmosphere',
                'noise', 'seating', 'parking', 'delivery', 'quality'
            },
            'hotels': {
                'room', 'staff', 'location', 'cleanliness', 'price',
                'amenities', 'breakfast', 'wifi', 'bed', 'bathroom',
                'pool', 'parking', 'checkin', 'check-in', 'view',
                'noise', 'towel', 'shower', 'lobby', 'service', 'gym'
            }
        }

        # Opinion words for context
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'perfect', 'love', 'best', 'nice', 'clean', 'comfortable',
            'fast', 'quick', 'helpful', 'friendly', 'fresh', 'delicious',
            'stunning', 'spacious', 'smooth', 'impressive', 'outstanding',
            'superb', 'brilliant', 'incredible', 'convenient', 'prompt'
        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst',
            'slow', 'dirty', 'rude', 'broken', 'noisy', 'expensive',
            'small', 'old', 'cheap', 'weak', 'dim', 'bland', 'stale',
            'disappointing', 'overpriced', 'outdated', 'cold', 'smelled',
            'unhelpful', 'crowded', 'uncomfortable', 'cramped', 'late'
        }

    def _get_sentiment(self, tokens, aspect_idx, window=4):
        """
        Determine sentiment for aspect at aspect_idx
        by looking at surrounding opinion words.
        """
        start = max(0, aspect_idx - window)
        end = min(len(tokens), aspect_idx + window + 1)
        context = [t.lower().rstrip('.,!?') for t in tokens[start:end]]

        pos_count = sum(1 for t in context if t in self.positive_words)
        neg_count = sum(1 for t in context if t in self.negative_words)

        # Negation detection
        negations = {'not', 'no', 'never', "wasn't", "isn't", "didn't", "don't", "couldn't"}
        negated = any(t in negations for t in context)

        if pos_count > neg_count:
            return 'negative' if negated else 'positive'
        elif neg_count > pos_count:
            return 'positive' if negated else 'negative'
        return 'neutral'

    def predict(self, text, domain):
        """
        Extract aspects from text for a given domain.
        Returns list of {term, polarity, start_token, end_token}
        """
        import re
        tokens = re.findall(r'\S+', text)
        tokens_clean = [t.lower().rstrip('.,!?;:') for t in tokens]

        indicators = self.aspect_indicators.get(domain, set())
        aspects = []
        seen = set()

        # Get feature weights if weighter available
        if self.weighter:
            try:
                weights = self.weighter.compute_token_weights(tokens, domain)
            except Exception:
                weights = [1.0] * len(tokens)
        else:
            weights = [1.0] * len(tokens)

        i = 0
        while i < len(tokens_clean):
            token = tokens_clean[i]

            # Check bigram first (e.g., "battery life", "wait time")
            bigram = None
            if i + 1 < len(tokens_clean):
                bigram = token + ' ' + tokens_clean[i + 1]

            # Seed match or high weight token
            is_aspect = (
                token in indicators or
                (bigram and any(
                    bigram in seed.lower() or seed.lower() in bigram
                    for seed in indicators
                )) or
                weights[i] > 0.6
            )

            if is_aspect and token not in seen and len(token) > 2:
                # Determine span
                if bigram and i + 1 < len(tokens_clean) and tokens_clean[i + 1] in indicators:
                    term = tokens[i] + ' ' + tokens[i + 1]
                    end_idx = i + 1
                else:
                    term = tokens[i]
                    end_idx = i

                polarity = self._get_sentiment(tokens, i)
                aspects.append({
                    'term': term,
                    'polarity': polarity,
                    'start_token': i,
                    'end_token': end_idx,
                    'weight': round(weights[i], 3)
                })
                seen.add(token)
                i = end_idx + 1
            else:
                i += 1

        return aspects

    def predict_bio(self, text, domain):
        """
        Returns BIO labels for each token.
        """
        import re
        tokens = re.findall(r'\S+', text)
        aspects = self.predict(text, domain)

        aspect_spans = set()
        for asp in aspects:
            aspect_spans.add(asp['start_token'])
            if asp['end_token'] != asp['start_token']:
                aspect_spans.add(asp['end_token'])

        bio = []
        for i in range(len(tokens)):
            if i in aspect_spans:
                prev_in = (i - 1) in aspect_spans
                bio.append('B-ASP' if not prev_in else 'I-ASP')
            else:
                bio.append('O')

        return tokens, bio

    def save(self, path=None):
        if path is None:
            path = os.path.join(Config.MODEL_DIR, 'rule_model.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[MODEL] Rule-based model saved to {path}")

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = os.path.join(Config.MODEL_DIR, 'rule_model.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════
# SECTION C: Unified Predictor Interface
# ═══════════════════════════════════════════════

class AspectPredictor:
    """
    Unified prediction interface.
    Uses BERT model if available, falls back to rule-based.
    """

    def __init__(self, weighter: FeatureWeightComputer = None):
        self.weighter = weighter
        self.model = None
        self.tokenizer = None
        self.mode = None
        self._load()

    def _load(self):
        if TORCH_AVAILABLE:
            try:
                self.model, self.tokenizer = load_model()
                self.mode = 'bert'
                print("[PREDICTOR] Using BERT model")
                return
            except FileNotFoundError:
                pass

        # Try rule model
        try:
            self.model = RuleBasedAspectModel.load()
            self.mode = 'rule'
            print("[PREDICTOR] Using saved rule-based model")
            return
        except (FileNotFoundError, Exception):
            pass

        # Fresh rule model
        self.model = RuleBasedAspectModel(self.weighter)
        self.mode = 'rule'
        print("[PREDICTOR] Using fresh rule-based model")

    def predict(self, text, domain):
        """
        Extract aspects from text.
        Returns list of {term, polarity, weight}
        """
        if self.mode == 'bert':
            return self._bert_predict(text, domain)
        return self.model.predict(text, domain)

    def _bert_predict(self, text, domain):
        import re, torch
        device = next(self.model.parameters()).device
        tokens = re.findall(r'\S+', text)

        # Get weights
        weights = [1.0] * len(tokens)
        if self.weighter:
            try:
                weights = self.weighter.compute_token_weights(tokens, domain)
            except Exception:
                pass

        encoding = self.tokenizer(
            tokens, is_split_into_words=True,
            max_length=Config.MAX_SEQ_LENGTH,
            padding='max_length', truncation=True,
            return_tensors='pt'
        )
        word_ids = encoding.word_ids()

        # Align weights to subword tokens
        aligned_weights = []
        prev = None
        for wid in word_ids:
            if wid is None:
                aligned_weights.append(0.0)
            elif wid != prev:
                aligned_weights.append(weights[wid] if wid < len(weights) else 1.0)
            else:
                aligned_weights.append(aligned_weights[-1])
            prev = wid
        aligned_weights = (aligned_weights + [0.0] * Config.MAX_SEQ_LENGTH)[:Config.MAX_SEQ_LENGTH]

        domain_id = torch.tensor([DOMAIN_LABEL2ID.get(domain, 0)])
        token_weights_t = torch.tensor([aligned_weights], dtype=torch.float)

        with torch.no_grad():
            logits = self.model(
                encoding['input_ids'].to(device),
                encoding['attention_mask'].to(device),
                domain_id.to(device),
                token_weights_t.to(device)
            )
        preds = torch.argmax(logits, dim=-1).squeeze().tolist()

        # Decode predictions back to word level
        aspects = []
        current_aspect_tokens = []
        prev_wid = None

        for pos, wid in enumerate(word_ids):
            if wid is None or pos >= len(preds):
                if current_aspect_tokens:
                    term = ' '.join(current_aspect_tokens)
                    aspects.append({
                        'term': term,
                        'polarity': 'neutral',
                        'weight': 1.0
                    })
                    current_aspect_tokens = []
                continue
            if wid == prev_wid:
                continue
            label_id = preds[pos]
            label = BIO_ID2LABEL.get(label_id, 'O')
            token = tokens[wid] if wid < len(tokens) else ''
            if label == 'B-ASP':
                if current_aspect_tokens:
                    aspects.append({'term': ' '.join(current_aspect_tokens), 'polarity': 'neutral', 'weight': 1.0})
                current_aspect_tokens = [token]
            elif label == 'I-ASP' and current_aspect_tokens:
                current_aspect_tokens.append(token)
            else:
                if current_aspect_tokens:
                    aspects.append({'term': ' '.join(current_aspect_tokens), 'polarity': 'neutral', 'weight': 1.0})
                    current_aspect_tokens = []
            prev_wid = wid

        return aspects


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("Module 3: Aspect Extraction Model")
    print("=" * 55)

    # Load data
    try:
        dataset, splits = load_processed()
    except FileNotFoundError:
        dataset, stats = build_dataset()
        splits = split_dataset(dataset)
        save_processed(dataset, splits)

    # Load weighter
    try:
        weighter = FeatureWeightComputer.load()
    except FileNotFoundError:
        weighter = FeatureWeightComputer().fit(dataset)
        weighter.save()

    # Add weights to splits
    for domain in splits:
        for split_name in ('train', 'val'):
            splits[domain][split_name] = [
                weighter.compute_record_weights(r)
                for r in splits[domain][split_name]
            ]

    if TORCH_AVAILABLE:
        print("\n[MODEL] Training BERT model...")
        model, tokenizer, history = train_model(splits, weighter)
        print(f"\n[HISTORY] Final Val F1: {history['val_f1'][-1]:.4f}")
    else:
        print("\n[MODEL] Building rule-based model (PyTorch not available)...")
        model = RuleBasedAspectModel(weighter)
        model.save()

    # Test predictions
    print("\n[TEST] Aspect Predictions:")
    predictor = AspectPredictor(weighter)

    test_cases = [
        ("The battery life is amazing but the screen is too dim.", "electronics"),
        ("Service was slow but the food tasted absolutely incredible.", "restaurants"),
        ("The room was spacious but wifi was terrible and staff were rude.", "hotels"),
        ("Camera quality is stunning and the price is very reasonable.", "electronics"),
    ]

    for text, domain in test_cases:
        aspects = predictor.predict(text, domain)
        print(f"\n  [{domain}] \"{text}\"")
        for asp in aspects:
            print(f"    → '{asp['term']}' [{asp['polarity']}] (weight={asp.get('weight', '-')})")

    print("\n✅ Module 3 complete.")
