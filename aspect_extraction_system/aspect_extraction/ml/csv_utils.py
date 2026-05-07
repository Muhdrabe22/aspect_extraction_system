"""
Module 9: CSV Upload & Export Utilities
Handles robust CSV parsing, validation, batch export,
and SemEval dataset downloader instructions.
"""

import os
import sys
import io
import csv
import json
import re
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


# ─────────────────────────────────────────────
# 1. CSV Parser / Validator
# ─────────────────────────────────────────────

class CSVParser:
    """
    Robust CSV parser for review uploads.
    Handles various encodings, delimiters, column names.
    """

    REVIEW_COLUMN_NAMES = {
        'review', 'text', 'comment', 'body', 'content',
        'review_text', 'reviewtext', 'review text',
        'feedback', 'opinion', 'description', 'message',
    }

    MAX_REVIEWS = 500
    MAX_TEXT_LENGTH = 2000
    MIN_TEXT_LENGTH = 10

    def parse(self, file_content, filename='upload.csv'):
        """
        Parse CSV/TXT file content into list of review strings.

        Returns:
            {
                'reviews': [str],
                'total': int,
                'skipped': int,
                'errors': [str],
                'columns': [str],
                'format': str   # 'csv' | 'txt'
            }
        """
        errors = []
        reviews = []
        skipped = 0

        # Detect format
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'csv'

        if ext == 'txt':
            return self._parse_txt(file_content)

        # Try CSV parsing
        try:
            # Detect delimiter
            sample = file_content[:2000]
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ','

        try:
            reader = csv.DictReader(io.StringIO(file_content), delimiter=delimiter)
            fieldnames = [f.strip().lower() for f in (reader.fieldnames or [])]

            # Find review column
            text_col = None
            original_col = None
            for i, col in enumerate(fieldnames):
                if col in self.REVIEW_COLUMN_NAMES:
                    text_col = col
                    original_col = (reader.fieldnames or [])[i]
                    break

            if not text_col and fieldnames:
                # Use first column
                text_col = fieldnames[0]
                original_col = (reader.fieldnames or [])[0]
                errors.append(f"No recognized review column. Using first column: '{original_col}'")

            if not text_col:
                return {'reviews': [], 'total': 0, 'skipped': 0,
                        'errors': ['No columns found in CSV'], 'columns': [], 'format': 'csv'}

            for row in reader:
                text = row.get(original_col, '').strip()

                # Validation
                if not text:
                    skipped += 1
                    continue
                if len(text) < self.MIN_TEXT_LENGTH:
                    skipped += 1
                    continue
                if len(text) > self.MAX_TEXT_LENGTH:
                    text = text[:self.MAX_TEXT_LENGTH]

                # Clean text
                text = self._clean_text(text)
                if text:
                    reviews.append(text)

                if len(reviews) >= self.MAX_REVIEWS:
                    errors.append(f"Truncated at {self.MAX_REVIEWS} reviews (max limit)")
                    break

        except Exception as e:
            errors.append(f"CSV parse error: {e}")

        return {
            'reviews': reviews,
            'total': len(reviews),
            'skipped': skipped,
            'errors': errors,
            'columns': fieldnames,
            'format': 'csv',
        }

    def _parse_txt(self, content):
        """Parse plain text file — one review per line."""
        lines = content.splitlines()
        reviews = []
        skipped = 0

        for line in lines:
            text = line.strip().strip('"\'')
            if not text or len(text) < self.MIN_TEXT_LENGTH:
                skipped += 1
                continue
            text = self._clean_text(text)
            if text:
                reviews.append(text[:self.MAX_TEXT_LENGTH])
            if len(reviews) >= self.MAX_REVIEWS:
                break

        return {
            'reviews': reviews,
            'total': len(reviews),
            'skipped': skipped,
            'errors': [],
            'columns': ['text'],
            'format': 'txt',
        }

    def _clean_text(self, text):
        """Basic text cleaning."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove null bytes
        text = text.replace('\x00', '')
        # Unescape common HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&#39;', "'").replace('&quot;', '"')
        return text

    def validate_domain(self, domain):
        """Validate domain string."""
        if domain not in Config.DOMAINS:
            return False, f"Invalid domain '{domain}'. Choose from: {Config.DOMAINS}"
        return True, None


# ─────────────────────────────────────────────
# 2. Export Builder
# ─────────────────────────────────────────────

class ExportBuilder:
    """
    Builds export files from analysis results.
    Supports CSV, JSON, and summary formats.
    """

    def build_aspects_csv(self, reviews_with_aspects):
        """
        Build detailed CSV of all aspects across reviews.
        reviews_with_aspects: list of Review ORM objects
        """
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            'review_id', 'review_text', 'domain',
            'aspect_term', 'polarity', 'score',
            'confidence', 'category', 'analyzed_at'
        ])

        for review in reviews_with_aspects:
            for asp in review.aspects:
                writer.writerow([
                    review.id,
                    review.text[:300].replace('\n', ' '),
                    review.domain,
                    asp.term,
                    asp.polarity,
                    round(asp.score, 4),
                    round(asp.confidence, 4),
                    asp.category,
                    review.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                ])

        output.seek(0)
        return output.getvalue()

    def build_summary_csv(self, batch_summary, job_id):
        """Build summary-level CSV for a batch job."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header info
        writer.writerow(['AspectLens Batch Export'])
        writer.writerow(['Job ID', job_id])
        writer.writerow(['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow(['Total Reviews', batch_summary.get('total_reviews', 0)])
        writer.writerow(['Overall Sentiment', batch_summary.get('overall_sentiment', '')])
        writer.writerow(['Positive Ratio', f"{batch_summary.get('positive_ratio', 0):.1%}"])
        writer.writerow([])

        # Top aspects
        writer.writerow(['--- TOP ASPECTS ---'])
        writer.writerow(['Aspect', 'Mentions', 'Dominant Sentiment', 'Avg Score', 'Category'])
        for asp in batch_summary.get('top_aspects', []):
            writer.writerow([
                asp.get('term', ''),
                asp.get('mention_count', 0),
                asp.get('dominant_polarity', ''),
                round(asp.get('avg_score', 0), 4),
                asp.get('category', ''),
            ])
        writer.writerow([])

        # Category breakdown
        writer.writerow(['--- CATEGORY BREAKDOWN ---'])
        writer.writerow(['Category', 'Total', 'Positive', 'Negative', 'Neutral'])
        for cat, stats in batch_summary.get('category_breakdown', {}).items():
            writer.writerow([
                cat,
                stats.get('total', 0),
                stats.get('positive', 0),
                stats.get('negative', 0),
                stats.get('neutral', 0),
            ])

        output.seek(0)
        return output.getvalue()

    def build_json_export(self, reviews_with_aspects):
        """Build JSON export of all analysis data."""
        data = {
            'exported_at': datetime.now().isoformat(),
            'total_reviews': len(reviews_with_aspects),
            'reviews': []
        }

        for review in reviews_with_aspects:
            data['reviews'].append({
                'id': review.id,
                'text': review.text,
                'domain': review.domain,
                'analyzed_at': review.created_at.isoformat(),
                'aspects': [
                    {
                        'term': asp.term,
                        'polarity': asp.polarity,
                        'score': asp.score,
                        'confidence': asp.confidence,
                        'category': asp.category,
                    }
                    for asp in review.aspects
                ]
            })

        return json.dumps(data, indent=2)

    def build_sample_csv(self, domain='restaurants'):
        """Generate a sample CSV for the given domain to show users the expected format."""
        samples = {
            'restaurants': [
                "The food was absolutely delicious and portions were very generous.",
                "Service was extremely slow and staff were rude and unhelpful.",
                "Amazing ambiance, cozy atmosphere. Prices are quite reasonable.",
                "The menu has great variety but wait time was over an hour.",
                "Best dessert I've had in years! Location is hard to find though.",
            ],
            'electronics': [
                "Battery life is incredible, easily lasts two full days.",
                "The screen is too dim for outdoor use, very disappointing.",
                "Camera quality is stunning and performance is blazing fast.",
                "Build quality feels cheap but the price is very affordable.",
                "Software updates are smooth. Charging speed is frustratingly slow.",
            ],
            'hotels': [
                "The room was spotless and the view was absolutely stunning.",
                "Staff were incredibly friendly and helpful throughout our stay.",
                "Wifi was terrible and parking was very expensive.",
                "Breakfast was excellent with a huge variety of options.",
                "The bed was so comfortable. Noise from street was a problem though.",
            ],
        }

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['review'])
        for text in samples.get(domain, samples['restaurants']):
            writer.writerow([text])
        output.seek(0)
        return output.getvalue()


# ─────────────────────────────────────────────
# 3. SemEval Dataset Instructions
# ─────────────────────────────────────────────

SEMEVAL_INSTRUCTIONS = """
# Downloading SemEval 2014 Task 4 Dataset

To train the BERT model on real data, download the SemEval 2014 dataset:

## Step 1: Download
Visit: https://alt.qcri.org/semeval2014/task4/
Or use the direct links:
  - Restaurants: https://alt.qcri.org/semeval2014/task4/data/uploads/semeval14-absa-train-v2.0.zip
  - Laptops:     https://alt.qcri.org/semeval2014/task4/data/uploads/semeval14-absa-train-v2.0.zip

## Step 2: Place files
Extract and place XML files in: data/semeval/

  data/semeval/
  ├── Restaurants_Train_v2.xml
  └── Laptop_Train_v2.xml

## Step 3: Rebuild dataset
python ml/dataset.py

## Expected file format (SemEval XML):
<sentences>
  <sentence id="...">
    <text>The food is good.</text>
    <aspectTerms>
      <aspectTerm term="food" polarity="positive" from="4" to="8"/>
    </aspectTerms>
  </sentence>
</sentences>

Note: The system works without SemEval files using synthetic data.
For best BERT model accuracy, use the full SemEval dataset.
"""


# ─────────────────────────────────────────────
# 4. Flask route additions for export
# ─────────────────────────────────────────────

def get_exporter():
    return ExportBuilder()

def get_parser():
    return CSVParser()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("Module 9: CSV Upload & Export")
    print("=" * 55)

    parser = CSVParser()
    exporter = ExportBuilder()

    # ── Test 1: CSV parsing
    print("\n[TEST 1] CSV Parsing:")
    sample_csv = """review,rating
"The battery life is amazing and screen is beautiful.",5
"Service was slow but food tasted incredible.",3
"Room was spotless. Wifi was terrible.",2
"",1
"Hi",1
"""
    result = parser.parse(sample_csv, 'reviews.csv')
    print(f"  Parsed:  {result['total']} reviews")
    print(f"  Skipped: {result['skipped']} (empty/too short)")
    print(f"  Errors:  {result['errors']}")
    for r in result['reviews']:
        print(f"  → '{r[:60]}'")

    # ── Test 2: TXT parsing
    print("\n[TEST 2] TXT Parsing:")
    sample_txt = """The camera quality is absolutely stunning.
Service was rude and food was cold and stale.
Excellent location but terrible wifi.
"""
    result2 = parser.parse(sample_txt, 'reviews.txt')
    print(f"  Parsed: {result2['total']} reviews from TXT")

    # ── Test 3: Sample CSV generation
    print("\n[TEST 3] Sample CSV Generation:")
    for domain in Config.DOMAINS:
        csv_data = exporter.build_sample_csv(domain)
        lines = csv_data.strip().splitlines()
        print(f"  {domain}: {len(lines)-1} sample rows")

    # ── Test 4: Export builder (mock data)
    print("\n[TEST 4] Export Builder:")

    class MockAsp:
        def __init__(self, term, polarity, score, confidence, category):
            self.term=term; self.polarity=polarity; self.score=score
            self.confidence=confidence; self.category=category

    class MockReview:
        def __init__(self, id, text, domain):
            self.id=id; self.text=text; self.domain=domain
            self.created_at=datetime.now()
            self.aspects=[
                MockAsp('battery','positive',0.8,0.9,'performance'),
                MockAsp('screen','negative',-0.6,0.85,'comfort'),
            ]

    mock_reviews = [
        MockReview(1, "Battery is great but screen too dim.", "electronics"),
        MockReview(2, "Food was amazing. Service terrible.", "restaurants"),
    ]
    csv_out = exporter.build_aspects_csv(mock_reviews)
    lines = csv_out.strip().splitlines()
    print(f"  Aspects CSV: {len(lines)} rows (incl. header)")

    json_out = exporter.build_json_export(mock_reviews)
    data = json.loads(json_out)
    print(f"  JSON export: {data['total_reviews']} reviews, "
          f"{sum(len(r['aspects']) for r in data['reviews'])} aspects")

    print("\n[INFO] SemEval Dataset:")
    print(SEMEVAL_INSTRUCTIONS[:300] + "...")

    print("\n✅ Module 9 complete.")
