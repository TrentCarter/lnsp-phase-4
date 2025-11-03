#!/usr/bin/env python3

"""
LVM Evaluation Dashboard
A Flask web application for evaluating LVM models
"""

import os
import sys

# Add the parent directory to the path so we can import from the lvm_eval package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from lvm_eval import app

if __name__ == '__main__':
    # Run the Flask development server
    app.run(
        host='0.0.0.0',
        port=8999,
        debug=True,
        threaded=True
    )
