import logging
import os

def setup_logger():

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join("logs/execution.log"),
        filemode='w'
    )