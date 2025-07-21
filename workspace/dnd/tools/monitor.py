import os
import time
from _thread import start_new_thread


def start_monitor(second=2):
    def _monitor():
        while True:
            time.sleep(second)
            os.system("nvidia-smi")

    start_new_thread(_monitor, ())
