"""
Shared helper functions used across the pipeline:
logging setup and the manifest writer that tracks every image's journey.
"""

# Python's built-in logging module — lets us print messages to console
# AND save them to a file, with timestamps and severity levels.
import logging

# Used to write rows into the manifest CSV file.
import csv

# Used to generate a timestamp for each manifest entry.
from datetime import datetime

# Same as in config.py — used here for file existence checks.
from pathlib import Path

# Imports our config.py file so we can use LOG_DIR and MANIFEST_PATH
# instead of hardcoding paths again.
import config


#             -------- LOGGER SETUP ---------

# Defines a function that creates and returns a configured logger object. 
# "name" lets different files create loggers with different labels if needed.
def setupLogger(name="pipeline"):
    # Creates the logs/ folder if it doesn't already exist.
    config.log_dir.mkdir(parents=True, exist_ok=True)
    # Creates (or retrieves, if already created) a logger object with this name.
    logger = logging.getLogger(name)
    # Sets the minimum severity level this logger will record.
    logger.setLevel(logging.INFO)

    # Checks if this logger already has output handlers attached.
    # Prevents duplicate log lines if setup_logger() gets called more than once.
    if not logger.handlers:
        # Defines the format of each log line: timestamp - severity level - the actual message.
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        # Creates a handler that prints log messages to the terminal.
        consoleHandler = logging.StreamHandler()
        # Applies our format to the console output.
        consoleHandler.setFormatter(formatter)
        # Creates a handler that writes log messages to a file called pipeline.log inside the logs/ folder.
        fileHandler = logging.FileHandler(config.log_dir / "pipeline.log")
        # Applies the same format to the file output.
        fileHandler.setFormatter(formatter)
        # Attaches the console handler to the logger.
        logger.addHandler(consoleHandler)
        # Attaches the file handler to the logger.
        logger.addHandler(fileHandler)
    # Returns the fully configured logger so other files can use it.
    return logger



#             -------- MANIFEST WRITER ---------

"""
A class to handle writing rows into the manifest CSV file.
Using a class (not just a function) lets us keep track of whether the header row has already been written.
"""
class ManifestWriter:
    # The constructor — runs once when you create a ManifestWriter object.
    def __init__(self, manifest_path=None):
        # Uses the provided path if given, otherwise falls back to the default path defined in config.py.
        self.manifest_path = manifest_path or config.manifest_dir
        # Makes sure the logs/ folder exists before we try to write into it.
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        # Calls a helper method (defined below) to add column headers only if the file doesn't already exist.
        self._write_header_if_needed()
    
    # A "private" helper method
    def _write_header_if_needed(self):
        # Checks whether the manifest file already exists on disk.
        if not self.manifest_path.exists():
            # Opens the file in write mode, creating it fresh. newline="" prevents extra blank lines on Windows.
            with open(self.manifest_path, mode="w", newline="") as f:
                # Creates a CSV writer object tied to this file.
                writer = csv.writer(f)
                # Writes the column headers as the first row.
                writer.writerow(["image_id", "source", "stage", "status", "timestamp", "notes"])

    # The main method other files will call to record an event.
    # Example: log_entry("img_001", "disk", "validation", "rejected", "corrupt file")
    def log_entry(self, image_id, source, stage, status, notes=""):
        # Opens the file in append mode, so we add to it without erasing previous entries.
        with open(self.manifest_path, mode="a", newline="") as f:
            # Creates a CSV writer object for this append operation.
            writer = csv.writer(f)
            # Writes one row: the image's ID, where it came from, which pipeline stage this is, the result, the current timestamp,
            # and any extra notes (like a rejection reason).
            writer.writerow([image_id, source, stage, status, datetime.now().isoformat(), notes])


#             -------- BATCH MANIFEST WRITER ---------

# Collects manifest entries in memory and writes them all at once in a single file open/write/close operation per flush.
class BatchManifestWriter():
    def __init__(self, manifestPath=None):
        # Path to the same manifest CSV the regular writer uses
        self.manifestPath = manifestPath or config.manifest_dir
        # Ensures the logs/ folder exists
        self.manifestPath.parent.mkdir(parents=True, exist_ok=True)
        # In-memory list of rows waiting to be written
        self._entries = []
        # Writes column headers if the file doesn't exist yet
        self._write_header_if_needed()

    def _write_header_if_needed(self):
        if not self.manifestPath.exists():
            with open(self.manifestPath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image_id", "source", "stage", "status", "timestamp", "notes"])
    
    # Adds one row to the in-memory buffer — no file I/O at all
    def addEntry(self, imageID, source, stage, status, notes=""):
        # Appends the row data with a timestamp to the in-memory list
        self._entries.append([
            imageID, source, stage, status, 
            datetime.now().isoformat(), notes
        ])
    
    # Writes ALL buffered entries to disk in one single file operation
    def flush(self):
        if not self._entries:
            return
        # writerows() writes the entire list in one call — far faster than calling writerow() in a loop
        with open(self.manifestPath, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self._entries)
        # Clears the buffer after writing so entries aren't duplicated on the next flush call
        self._entries.clear()


#             -------- QUICK TEST BLOCK ---------

# This block only runs if you execute "python utils.py" directly — it won't run if this file is just imported by another file.
if __name__ == "__main__":
    # Creates a logger using our function above.
    logger = setupLogger()
    # Logs a test message — should appear both in the terminal and in pipeline.log.
    logger.info("Testing logger setup")
    # Creates a ManifestWriter object using the default path from config.py.
    manifest = ManifestWriter()
    # Writes one test row into manifest.csv to confirm everything works.
    manifest.log_entry("test_001", "disk", "validation", "passed", "no issues found")
    # Logs a confirmation message.
    logger.info("Test entry written to manifest.csv")






