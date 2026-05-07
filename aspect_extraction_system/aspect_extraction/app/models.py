"""
Database Models — SQLAlchemy ORM
Tables: Review, AspectResult, BatchJob
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Review(db.Model):
    __tablename__ = 'reviews'

    id          = db.Column(db.Integer, primary_key=True)
    text        = db.Column(db.Text, nullable=False)
    domain      = db.Column(db.String(50), nullable=False)
    source      = db.Column(db.String(20), default='single')   # 'single' | 'csv'
    batch_id    = db.Column(db.Integer, db.ForeignKey('batch_jobs.id'), nullable=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    aspects     = db.relationship('AspectResult', backref='review',
                                  cascade='all, delete-orphan', lazy=True)

    def to_dict(self):
        return {
            'id':         self.id,
            'text':       self.text,
            'domain':     self.domain,
            'source':     self.source,
            'batch_id':   self.batch_id,
            'created_at': self.created_at.isoformat(),
            'aspects':    [a.to_dict() for a in self.aspects],
        }


class AspectResult(db.Model):
    __tablename__ = 'aspect_results'

    id           = db.Column(db.Integer, primary_key=True)
    review_id    = db.Column(db.Integer, db.ForeignKey('reviews.id'), nullable=False)
    term         = db.Column(db.String(100), nullable=False)
    polarity     = db.Column(db.String(20), nullable=False)
    score        = db.Column(db.Float, default=0.0)
    confidence   = db.Column(db.Float, default=0.0)
    category     = db.Column(db.String(50), default='other')
    weight       = db.Column(db.Float, default=0.0)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':         self.id,
            'review_id':  self.review_id,
            'term':       self.term,
            'polarity':   self.polarity,
            'score':      self.score,
            'confidence': self.confidence,
            'category':   self.category,
            'weight':     self.weight,
        }


class BatchJob(db.Model):
    __tablename__ = 'batch_jobs'

    id              = db.Column(db.Integer, primary_key=True)
    filename        = db.Column(db.String(255), nullable=False)
    domain          = db.Column(db.String(50), nullable=False)
    status          = db.Column(db.String(20), default='pending')  # pending|processing|done|failed
    total_reviews   = db.Column(db.Integer, default=0)
    processed       = db.Column(db.Integer, default=0)
    overall_sentiment = db.Column(db.String(20), nullable=True)
    positive_ratio  = db.Column(db.Float, default=0.0)
    summary_json    = db.Column(db.Text, nullable=True)   # JSON blob
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at    = db.Column(db.DateTime, nullable=True)

    reviews         = db.relationship('Review', backref='batch',
                                      foreign_keys='Review.batch_id', lazy=True)

    def to_dict(self):
        import json
        return {
            'id':               self.id,
            'filename':         self.filename,
            'domain':           self.domain,
            'status':           self.status,
            'total_reviews':    self.total_reviews,
            'processed':        self.processed,
            'overall_sentiment': self.overall_sentiment,
            'positive_ratio':   self.positive_ratio,
            'summary':          json.loads(self.summary_json) if self.summary_json else None,
            'created_at':       self.created_at.isoformat(),
            'completed_at':     self.completed_at.isoformat() if self.completed_at else None,
        }
