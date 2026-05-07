# AspectLens
### Aspect Extraction in Product Reviews Using Multi-Domain Feature Weighting and Knowledge Transfer

A full-stack NLP web application that extracts aspects from product/service reviews,
classifies their sentiment, and transfers knowledge across domains (Electronics, Restaurants, Hotels).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python run.py

# 3. Open browser
http://localhost:5000
```

---

## Project Structure

```
aspect_extraction/
├── run.py                    ← Entry point
├── config.py                 ← All settings
├── requirements.txt
│
├── app/                      ← Flask web layer
│   ├── __init__.py           ← App factory
│   ├── routes.py             ← All 18 routes (pages + API)
│   ├── models.py             ← SQLite ORM (Review, AspectResult, BatchJob)
│   ├── pipeline.py           ← Singleton ML pipeline loader
│   └── templates/
│       ├── base.html         ← Design system + sidebar
│       ├── index.html        ← Dashboard
│       ├── analyze.html      ← Single review analysis
│       ├── batch.html        ← CSV upload
│       ├── result.html       ← Result detail
│       ├── history.html      ← Review history
│       └── batch_result.html ← Batch report
│
├── ml/                       ← Machine Learning modules
│   ├── dataset.py            ← Module 1: Data pipeline + SemEval parser
│   ├── feature_weighting.py  ← Module 2: Multi-domain TF-IDF + seed boosting
│   ├── model.py              ← Module 3: BERT model + rule-based fallback
│   ├── sentiment.py          ← Module 4: Per-aspect sentiment classifier
│   ├── knowledge_transfer.py ← Module 5: Cross-domain transfer engine
│   ├── summary.py            ← Module 6: Template + LLM summary generator
│   └── csv_utils.py          ← Module 9: CSV parsing + export builder
│
├── data/semeval/             ← Place SemEval XML files here
├── saved_models/             ← Trained model weights saved here
└── uploads/                  ← Temporary CSV upload storage
```

---

## System Architecture

```
Review Text + Domain
        ↓
[Module 2] Feature Weighting
  Domain-aware TF-IDF + Seed Boost + Cross-domain transfer weights
        ↓
[Module 3] Aspect Extraction
  BERT fine-tuned (BIO tagging) OR Rule-based fallback
        ↓
[Module 4] Sentiment Classification
  Lexicon + intensifiers + negation + context window scoring
        ↓
[Module 5] Knowledge Transfer
  Taxonomy mapping + pivot features + domain adaptation
        ↓
[Module 6] Summary Generation
  Template summary + optional Claude API narrative
        ↓
Flask API → SQLite → Web UI
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard |
| GET | `/analyze` | Single review page |
| GET | `/batch` | CSV upload page |
| GET | `/history` | Review history |
| POST | `/api/analyze` | Analyze single review |
| POST | `/api/batch` | Upload + process CSV |
| GET | `/api/batch/<id>` | Batch job status |
| GET | `/api/stats` | System statistics |
| GET | `/api/export/csv/<id>` | Export batch as CSV |
| GET | `/api/export/json/<id>` | Export batch as JSON |
| GET | `/api/export/summary/<id>` | Export summary CSV |
| GET | `/api/export/all` | Export all reviews |
| GET | `/api/sample-csv/<domain>` | Download sample CSV |
| DELETE | `/api/review/<id>` | Delete a review |

### POST /api/analyze

```json
// Request
{ "text": "Battery life is amazing but screen is dim.", "domain": "electronics" }

// Response
{
  "aspects": [
    { "term": "battery", "polarity": "positive", "score": 0.572,
      "confidence": 0.9, "category": "performance" },
    { "term": "screen", "polarity": "negative", "score": -0.569,
      "confidence": 0.85, "category": "comfort" }
  ],
  "overall_sentiment": "mixed",
  "summary": { "short_summary": "...", "key_points": [...] }
}
```

### CSV Upload Format

```csv
review
"The battery life is incredible, easily lasts all day."
"Service was extremely slow and staff were rude."
"Room was spotless with a stunning view."
```

---

## Training the BERT Model

The system ships with a rule-based fallback model that works out of the box.
To train the full BERT model:

### 1. Download SemEval 2014 Dataset
```
https://alt.qcri.org/semeval2014/task4/
```
Place XML files in `data/semeval/`:
- `Restaurants_Train_v2.xml`
- `Laptop_Train_v2.xml`

### 2. Install PyTorch + Transformers
```bash
pip install torch transformers
```

### 3. Train
```bash
python ml/model.py
```
Training runs 3 epochs on CPU (~20 min) or GPU (~3 min).
Best model saved to `saved_models/aspect_model.pt`.

---

## Domains & Aspect Seeds

| Domain | Example Aspects |
|--------|----------------|
| Electronics | battery, screen, camera, performance, price, keyboard, speaker |
| Restaurants | food, service, ambiance, staff, price, menu, wait time |
| Hotels | room, staff, location, cleanliness, wifi, breakfast, parking |

---

## Knowledge Transfer

The system transfers knowledge across domains via:

1. **Aspect Taxonomy** — Maps domain-specific terms to 8 universal categories
   (quality, performance, price, staff, ambiance, comfort, connectivity, cleanliness)

2. **Pivot Features** — Shared vocabulary (price, location, quality, staff)
   acts as transfer anchors between domains

3. **Instance Weighting** — Source-domain training samples are scored by
   similarity to target domain for smarter cross-domain learning

---

## Tech Stack

- **Backend**: Python 3.10+, Flask 3.0, SQLAlchemy, SQLite
- **ML**: PyTorch, HuggingFace Transformers (BERT), scikit-learn
- **NLP**: Custom TF-IDF, lexicon-based sentiment, BIO tagging
- **Frontend**: Jinja2 templates, Chart.js, vanilla JS
- **Design**: Dark industrial — Bebas Neue + Space Mono + DM Sans

---

## Configuration (`config.py`)

```python
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_SEQ_LENGTH  = 128
BATCH_SIZE      = 16
LEARNING_RATE   = 2e-5
NUM_EPOCHS      = 3
DOMAINS         = ['electronics', 'restaurants', 'hotels']
```# aspect_extraction_system
Aspect Extraction in Product Reviews Using Multi-Domain Feature Weighting and Knowledge Transfer
