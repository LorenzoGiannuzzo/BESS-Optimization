import logging
import os

import logging
import os
def setup_logger():

    log_dir = "logs"
    log_file = os.path.join(log_dir, "execution.log")

    # Create the log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='w'
    )