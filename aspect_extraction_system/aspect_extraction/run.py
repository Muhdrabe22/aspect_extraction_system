"""Application entry point."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app

app = create_app()

if __name__ == '__main__':
    print("=" * 50)
    print(" Aspect Extraction System — Starting")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
