"""
Flask Routes — Module 7
REST API + Page endpoints

Pages:
  GET  /                    → Dashboard / home
  GET  /analyze             → Single review analysis page
  GET  /batch               → Batch/CSV upload page
  GET  /history             → Review history page
  GET  /result/<id>         → Single result detail page

API:
  POST /api/analyze         → Analyze single review
  POST /api/batch           → Upload + process CSV
  GET  /api/batch/<id>      → Batch job status
  GET  /api/history         → Recent reviews + results
  GET  /api/stats           → System statistics
  DELETE /api/review/<id>   → Delete a review
"""

import os
import re
import json
import csv
import io
from datetime import datetime, timedelta

from flask import (
    Blueprint, request, jsonify, render_template,
    redirect, url_for, flash, current_app
)
from werkzeug.utils import secure_filename

from app.models import db, Review, AspectResult, BatchJob
from app.pipeline import analyze_review, get_pipeline
from ml.csv_utils import ExportBuilder, CSVParser

_exporter = ExportBuilder()
_parser = CSVParser()

bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'csv', 'txt'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────
# Page Routes
# ─────────────────────────────────────────────

@bp.route('/')
def index():
    """Dashboard — show recent stats + activity."""
    total_reviews = Review.query.count()
    total_aspects = AspectResult.query.count()
    total_batches = BatchJob.query.count()

    recent = Review.query.order_by(Review.created_at.desc()).limit(5).all()

    # Sentiment distribution
    from sqlalchemy import func
    sentiment_data = db.session.query(
        AspectResult.polarity,
        func.count(AspectResult.id)
    ).group_by(AspectResult.polarity).all()
    sentiment_dist = {row[0]: row[1] for row in sentiment_data}

    # Domain distribution
    domain_data = db.session.query(
        Review.domain,
        func.count(Review.id)
    ).group_by(Review.domain).all()
    domain_dist = {row[0]: row[1] for row in domain_data}

    # Recent activity (last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    weekly_reviews = Review.query.filter(Review.created_at >= week_ago).count()

    return render_template('index.html',
        total_reviews=total_reviews,
        total_aspects=total_aspects,
        total_batches=total_batches,
        recent_reviews=recent,
        sentiment_dist=sentiment_dist,
        domain_dist=domain_dist,
        weekly_reviews=weekly_reviews,
        domains=current_app.config['DOMAINS'],
    )


@bp.route('/analyze')
def analyze_page():
    """Single review analysis page."""
    return render_template('analyze.html',
        domains=current_app.config['DOMAINS']
    )


@bp.route('/batch')
def batch_page():
    """Batch CSV upload page."""
    jobs = BatchJob.query.order_by(BatchJob.created_at.desc()).limit(10).all()
    return render_template('batch.html',
        domains=current_app.config['DOMAINS'],
        jobs=jobs
    )


@bp.route('/history')
def history_page():
    """Review history with pagination."""
    page = request.args.get('page', 1, type=int)
    domain_filter = request.args.get('domain', '')
    sentiment_filter = request.args.get('sentiment', '')

    query = Review.query.order_by(Review.created_at.desc())
    if domain_filter:
        query = query.filter_by(domain=domain_filter)

    reviews = query.paginate(page=page, per_page=20, error_out=False)

    return render_template('history.html',
        reviews=reviews,
        domains=current_app.config['DOMAINS'],
        domain_filter=domain_filter,
    )


@bp.route('/result/<int:review_id>')
def result_page(review_id):
    """Single result detail view."""
    review = Review.query.get_or_404(review_id)
    return render_template('result.html', review=review)


@bp.route('/batch/result/<int:job_id>')
def batch_result_page(job_id):
    """Batch job result detail page."""
    job = BatchJob.query.get_or_404(job_id)
    summary = json.loads(job.summary_json) if job.summary_json else {}
    return render_template('batch_result.html', job=job, summary=summary)


# ─────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────

@bp.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Analyze a single review.
    Body: {text: str, domain: str, use_llm: bool}
    Returns full analysis with aspects + sentiment + summary.
    """
    data = request.get_json(force=True)
    text   = (data.get('text') or '').strip()
    domain = (data.get('domain') or 'restaurants').lower()
    use_llm = data.get('use_llm', True)

    if not text:
        return jsonify({'error': 'Review text is required'}), 400
    if len(text) > 5000:
        return jsonify({'error': 'Review text too long (max 5000 chars)'}), 400
    if domain not in current_app.config['DOMAINS']:
        return jsonify({'error': f"Invalid domain. Choose from: {current_app.config['DOMAINS']}"}), 400

    try:
        result = analyze_review(text, domain, use_llm=use_llm)
    except Exception as e:
        current_app.logger.error(f"Analysis error: {e}")
        return jsonify({'error': 'Analysis failed', 'detail': str(e)}), 500

    # Persist to DB
    try:
        review = Review(text=text, domain=domain, source='single')
        db.session.add(review)
        db.session.flush()

        for asp in result['aspects']:
            ar = AspectResult(
                review_id  = review.id,
                term       = asp.get('term', '')[:100],
                polarity   = asp.get('polarity', 'neutral'),
                score      = asp.get('score', 0.0),
                confidence = asp.get('confidence', 0.0),
                category   = asp.get('category', 'other'),
                weight     = asp.get('weight', 0.0),
            )
            db.session.add(ar)

        db.session.commit()
        result['review_id'] = review.id

    except Exception as e:
        db.session.rollback()
        current_app.logger.warning(f"DB save failed: {e}")

    return jsonify(result)


@bp.route('/api/batch', methods=['POST'])
def api_batch():
    """
    Upload + process a CSV file of reviews.
    Form fields: file (CSV), domain (str)
    CSV format: one review per row, column named 'review' or 'text' or first column
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    domain = (request.form.get('domain') or 'restaurants').lower()

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only .csv and .txt files allowed'}), 400
    if domain not in current_app.config['DOMAINS']:
        return jsonify({'error': f"Invalid domain"}), 400

    # Read CSV
    try:
        content = file.read().decode('utf-8', errors='replace')
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            # Try as plain text (one review per line)
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            rows = [{'review': l} for l in lines]
    except Exception as e:
        return jsonify({'error': f'Could not parse file: {e}'}), 400

    if not rows:
        return jsonify({'error': 'File is empty'}), 400
    if len(rows) > 500:
        return jsonify({'error': 'Max 500 reviews per batch'}), 400

    # Find review text column
    fieldnames = list(rows[0].keys()) if rows else []
    text_col = next(
        (c for c in fieldnames if c.lower() in ('review', 'text', 'comment', 'body', 'content')),
        fieldnames[0] if fieldnames else None
    )
    if not text_col:
        return jsonify({'error': 'Could not find review text column'}), 400

    texts = [row.get(text_col, '').strip() for row in rows if row.get(text_col, '').strip()]
    if not texts:
        return jsonify({'error': 'No review texts found'}), 400

    # Create batch job
    job = BatchJob(
        filename=secure_filename(file.filename),
        domain=domain,
        status='processing',
        total_reviews=len(texts),
    )
    db.session.add(job)
    db.session.commit()

    # Process reviews
    review_results = []
    errors = 0

    for text in texts:
        try:
            result = analyze_review(text, domain, use_llm=False)

            # Save review
            review = Review(
                text=text[:2000],
                domain=domain,
                source='csv',
                batch_id=job.id,
            )
            db.session.add(review)
            db.session.flush()

            for asp in result['aspects']:
                ar = AspectResult(
                    review_id  = review.id,
                    term       = asp.get('term', '')[:100],
                    polarity   = asp.get('polarity', 'neutral'),
                    score      = asp.get('score', 0.0),
                    confidence = asp.get('confidence', 0.0),
                    category   = asp.get('category', 'other'),
                    weight     = asp.get('weight', 0.0),
                )
                db.session.add(ar)

            review_results.append(result)
            job.processed += 1

        except Exception as e:
            errors += 1
            current_app.logger.warning(f"Batch item error: {e}")

    # Build batch summary
    try:
        pipeline = get_pipeline()
        batch_summary = pipeline['summary'].summarize_batch(review_results, domain)
    except Exception as e:
        batch_summary = {}

    # Update job
    job.status = 'done'
    job.overall_sentiment = batch_summary.get('overall_sentiment', 'neutral')
    job.positive_ratio = batch_summary.get('positive_ratio', 0.0)
    job.summary_json = json.dumps(batch_summary)
    job.completed_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        'job_id': job.id,
        'status': 'done',
        'total': len(texts),
        'processed': job.processed,
        'errors': errors,
        'overall_sentiment': job.overall_sentiment,
        'positive_ratio': job.positive_ratio,
        'summary': batch_summary,
        'redirect': url_for('main.batch_result_page', job_id=job.id),
    })


@bp.route('/api/batch/<int:job_id>', methods=['GET'])
def api_batch_status(job_id):
    """Get batch job status and results."""
    job = BatchJob.query.get_or_404(job_id)
    return jsonify(job.to_dict())


@bp.route('/api/history', methods=['GET'])
def api_history():
    """Recent reviews with pagination."""
    page   = request.args.get('page', 1, type=int)
    domain = request.args.get('domain', '')
    limit  = min(request.args.get('limit', 20, type=int), 100)

    query = Review.query.order_by(Review.created_at.desc())
    if domain:
        query = query.filter_by(domain=domain)

    total = query.count()
    reviews = query.offset((page - 1) * limit).limit(limit).all()

    return jsonify({
        'total': total,
        'page': page,
        'limit': limit,
        'reviews': [r.to_dict() for r in reviews],
    })


@bp.route('/api/stats', methods=['GET'])
def api_stats():
    """System-wide statistics."""
    from sqlalchemy import func

    total_reviews  = Review.query.count()
    total_aspects  = AspectResult.query.count()
    total_batches  = BatchJob.query.filter_by(status='done').count()

    sentiment_dist = dict(db.session.query(
        AspectResult.polarity, func.count(AspectResult.id)
    ).group_by(AspectResult.polarity).all())

    domain_dist = dict(db.session.query(
        Review.domain, func.count(Review.id)
    ).group_by(Review.domain).all())

    category_dist = dict(db.session.query(
        AspectResult.category, func.count(AspectResult.id)
    ).group_by(AspectResult.category).all())

    top_aspects = db.session.query(
        AspectResult.term,
        func.count(AspectResult.id).label('count')
    ).group_by(AspectResult.term)\
     .order_by(func.count(AspectResult.id).desc())\
     .limit(10).all()

    return jsonify({
        'total_reviews':  total_reviews,
        'total_aspects':  total_aspects,
        'total_batches':  total_batches,
        'sentiment_dist': sentiment_dist,
        'domain_dist':    domain_dist,
        'category_dist':  category_dist,
        'top_aspects':    [{'term': t, 'count': c} for t, c in top_aspects],
    })


@bp.route('/api/review/<int:review_id>', methods=['DELETE'])
def api_delete_review(review_id):
    """Delete a review and its aspects."""
    review = Review.query.get_or_404(review_id)
    db.session.delete(review)
    db.session.commit()
    return jsonify({'deleted': True, 'id': review_id})


@bp.route('/api/export/json/<int:job_id>', methods=['GET'])
def api_export_json(job_id):
    """Export batch job results as JSON."""
    from flask import Response
    job = BatchJob.query.get_or_404(job_id)
    reviews = Review.query.filter_by(batch_id=job_id).all()
    data = _exporter.build_json_export(reviews)
    return Response(data, mimetype='application/json',
        headers={'Content-Disposition': f'attachment; filename=batch_{job_id}.json'})


@bp.route('/api/export/summary/<int:job_id>', methods=['GET'])
def api_export_summary(job_id):
    """Export batch summary CSV."""
    from flask import Response
    job = BatchJob.query.get_or_404(job_id)
    import json as _json
    summary = _json.loads(job.summary_json) if job.summary_json else {}
    data = _exporter.build_summary_csv(summary, job_id)
    return Response(data, mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=summary_{job_id}.csv'})


@bp.route('/api/sample-csv/<domain>', methods=['GET'])
def api_sample_csv(domain):
    """Download a sample CSV for a domain."""
    if domain not in current_app.config['DOMAINS']:
        return jsonify({'error': 'Invalid domain'}), 400
    data = _exporter.build_sample_csv(domain)
    from flask import Response
    return Response(data, mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=sample_{domain}.csv'})


@bp.route('/api/export/all', methods=['GET'])
def api_export_all():
    """Export all reviews as CSV."""
    from flask import Response
    reviews = Review.query.order_by(Review.created_at.desc()).limit(1000).all()
    data = _exporter.build_aspects_csv(reviews)
    fname = f'aspectlens_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    return Response(data, mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={fname}'})


@bp.route('/api/export/csv/<int:job_id>', methods=['GET'])
def api_export_csv(job_id):
    """Export batch job results as CSV."""
    from flask import Response
    job = BatchJob.query.get_or_404(job_id)
    reviews = Review.query.filter_by(batch_id=job_id).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['review_id', 'text', 'domain', 'aspect', 'polarity',
                     'score', 'confidence', 'category'])

    for review in reviews:
        for asp in review.aspects:
            writer.writerow([
                review.id, review.text[:200], review.domain,
                asp.term, asp.polarity, asp.score, asp.confidence, asp.category
            ])

    output.seek(0)
    filename = f"batch_{job_id}_results.csv"
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )
