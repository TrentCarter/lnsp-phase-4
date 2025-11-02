import sys
import os

# Add parent directory to path so we can import lvm_eval
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lvm_eval import app

if __name__ == '__main__':
    # Start the Flask application on port 8999
    print(f"Starting LVM Evaluation Dashboard on http://0.0.0.0:8999")
    app.run(debug=False, host='0.0.0.0', port=8999, threaded=True)
