import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'aspect-extraction-secret-key-2024')
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'aspect_extraction.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload

    # ML Config
    MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'semeval')
    BERT_MODEL_NAME = 'bert-base-uncased'
    MAX_SEQ_LENGTH = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3

    # Domains
    DOMAINS = ['electronics', 'restaurants', 'hotels']

    DOMAIN_ASPECT_SEEDS = {
        'electronics': [
            'battery', 'screen', 'camera', 'performance', 'price',
            'design', 'software', 'storage', 'display', 'keyboard',
            'speaker', 'charging', 'processor', 'memory', 'build quality'
        ],
        'restaurants': [
            'food', 'service', 'ambiance', 'price', 'menu',
            'staff', 'location', 'cleanliness', 'portion', 'taste',
            'drinks', 'dessert', 'wait time', 'reservation', 'atmosphere'
        ],
        'hotels': [
            'room', 'staff', 'location', 'cleanliness', 'price',
            'amenities', 'breakfast', 'wifi', 'bed', 'bathroom',
            'pool', 'parking', 'check-in', 'view', 'noise'
        ]
    }

    SENTIMENT_LABELS = ['positive', 'negative', 'neutral', 'conflict']
    BIO_LABELS = ['O', 'B-ASP', 'I-ASP']
