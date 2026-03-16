"""
Production WSGI entry point for gunicorn / uWSGI.
Usage: gunicorn wsgi:app -w 4 -b 0.0.0.0:5000
"""
from app import app

if __name__ == "__main__":
    app.run()
