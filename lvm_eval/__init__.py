from flask import Flask, session
import logging
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Configure session with a simple in-memory session
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configure paths
app.config['MODELS_DIR'] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    'artifacts', 'lvm', 'models'
)

# Create necessary directories
os.makedirs(app.config['MODELS_DIR'], exist_ok=True)
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Ensure logs directory exists
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'app.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Application initialized")

# Import routes after app is created to avoid circular imports
from . import routes

# Add datetime filter to format timestamps in templates
@app.template_filter('datetime')
def format_datetime(timestamp, format='%Y-%m-%d %H:%M:%S'):
    if timestamp is None:
        return ''
    if isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp).strftime(format)
    return timestamp.strftime(format) if hasattr(timestamp, 'strftime') else str(timestamp)

# Initialize session
@app.before_request
def make_session_permanent():
    session.permanent = True
