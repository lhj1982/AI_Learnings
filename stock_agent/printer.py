import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class Printer:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def heading(self, msg: str):
        if not self.verbose:
            return
        logging.info("=" * 80)
        logging.info(msg)
        logging.info("=" * 80)

    def step(self, msg: str):
        if not self.verbose:
            return
        logging.info(f"➡ {msg}")

    def warn(self, msg: str):
        if not self.verbose:
            return
        logging.warning(f"⚠ {msg}")
