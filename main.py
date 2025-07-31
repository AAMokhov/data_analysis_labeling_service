#!/usr/bin/env python3
"""
Main Entry Point for Data Analysis & Labeling Service
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory and app directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "app"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to run the application"""
    try:
        logger.info("Starting Data Analysis & Labeling Service")

        # Import and run the Dash app
        from app.dash_app import app

        logger.info("Dash application initialized successfully")
        logger.info("Starting web server on http://0.0.0.0:8050")

        # Run the app
        app.run(
            debug=False,  # Set to False for production
            host='0.0.0.0',
            port=8050,
            threaded=True
        )

    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
