import sys
import os

class Logger:
    def __init__(self, filename="logfile.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.log = open(filename, "a", encoding="utf-8")  # Specify utf-8 encoding

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# Usage
#log_file_path = os.path.join("Logs", "training_log.txt")
#sys.stdout = Logger(log_file_path)