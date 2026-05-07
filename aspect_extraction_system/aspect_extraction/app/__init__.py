"""Flask application factory."""
import os
from flask import Flask
from app.models import db
from config import Config


def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config.from_object(Config)

    os.makedirs(app.config.get('UPLOAD_FOLDER', 'uploads'), exist_ok=True)
    os.makedirs(app.config.get('MODEL_DIR', 'saved_models'), exist_ok=True)

    db.init_app(app)
    with app.app_context():
        db.create_all()

    from app.routes import bp
    app.register_blueprint(bp)

    return app
