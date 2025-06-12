import os
import logging

def setup_logger():
    log_dir = "logs"
    log_file = "execution.log"
    os.makedirs(log_dir, exist_ok=True)  # Create logs directory if it doesn't exist
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(log_dir, log_file),
        filemode='w'
    )