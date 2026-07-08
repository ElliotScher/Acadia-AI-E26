"""
Write load_from_disk(root_dir) — walks the directory, returns a list of valid file paths (filter by VALID_EXTENSIONS from config)
Write read_image(path) — uses cv2.imread(), returns None if the read fails (this is your first real corruption check, even before validation.py)
Test manually: point it at a small folder, print how many paths were found, confirm a few load as arrays with .shape
Only once disk works: stub out load_from_sql(query) — connect, fetch rows, return list of (id, path) tuples. Leave the actual DB credentials as a TODO until you have real DB access to test against
"""

"""
Handles loading images from different sources (disk, SQL database). Every function here returns data in a consistent format so the rest
of the pipeline never needs to know where an image actually came from.
"""

#OpenCV — used here specifically for cv2.imread(), which loads an image file from disk into a numpy array.
import cv2 as cv
# Lets us walk directories and check file extensions safely across OSes.
from pathlib import Path
# Imports our config.py so we can use VALID_EXTENSIONS and RAW_DIR instead of hardcoding them again here.
import config
# Imports our utils.py so we can use the logger to record what happens during ingestion (e.g. how many files were found, any read failures).
import utils
import numpy as np

#             -------- DISK INGESTION ---------

# Defines a function that scans a folder and returns valid image paths. root_dir is optional — if not given, falls back to config.RAW_DIR.
def loadFromDisk(root_dir=None):
    # Grabs our configured logger so we can record progress and issues.
    logger = utils.setupLogger()
    # If root_dir was passed in, wrap it in Path() to make sure it's a proper Path object. Otherwise, use the default from config.py.
    root_dir = Path(root_dir) if (root_dir) else config.raw_dir
    # Checks whether the folder we're about to scan actually exists.
    if not root_dir.exists():
        # Logs an error message with the missing path included.
        logger.error(f"Directory does not exist: {root_dir}")
        # Returns an empty list immediately — nothing to process.
        return []
    
    """
    This is a list comprehension that:
    - root_dir.rglob("*") recursively finds every file in every subfolder
    - p.suffix.lower() gets the file extension in lowercase (e.g. ".jpg")
    - keeps only files whose extension is in our allowed set from config.py
    """
    image_path = [
        p for p in root_dir.rglob("*")
        if p.suffix.lower() in config.valid_extensions
    ]
    # Logs how many matching files were found — useful sanity check every time you run this.
    logger.info(f"Found {len(image_path)} valid image files in {root_dir}")
    # Returns the list of Path objects pointing to valid image files.
    return image_path


#             -------- SQL INGESTION ---------

# Pulls image records from a SQL database using the configured credentials. Returns a list of (image_id, file_path, label) tuples
def loadFromSQL(query=None):
    logger = utils.setupLogger()
    # Checks that all four required credentials are actually set. all() returns False if any of them is None or empty
    if not all([config.dbHost, config.dbName, config.dbUser, config.dbPassword]):
        logger.error(
            "Missing database credentials. Set IMG_DB_HOST, IMG_DB_NAME, "
            "IMG_DB_USER, IMG_DB_PASSWORD as environment variables."
        )
        return []
    # Uses the custom query if one was passed in, otherwise uses the default query from config.py that filters for your three object classes
    query = query or config.dbImageQuery
    try:
        import pyodbc
        # Builds a connection string for Microsoft SQL Server.
        if config.dbType == "sqlserver":
            # The double braces {{ }} around DRIVER are needed because we're inside an f-string — they produce literal { } in the output
            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={config.DB_HOST},{config.DB_PORT};"
                f"DATABASE={config.DB_NAME};"
                f"UID={config.DB_USER};"
                f"PWD={config.DB_PASSWORD}"
            )
        # Builds a connection string for PostgreSQL
        elif config.dbType == "postgresql":
            connection_string = (
                f"DRIVER={{PostgreSQL Unicode}};"
                f"SERVER={config.DB_HOST};"
                f"PORT={config.DB_PORT};"
                f"DATABASE={config.DB_NAME};"
                f"UID={config.DB_USER};"
                f"PWD={config.DB_PASSWORD}"
            )
        else:
            logger.error(f"Unsupported DB_TYPE: {config.DB_TYPE}")
            return []
        # Logs the connection attempt before it actually happens
        logger.info(f"Connecting to {config.DB_TYPE} database: {config.DB_NAME}")
        # Opens the actual database connection using our connection string.
        connect = pyodbc.connect(connection_string)
        # Creates a cursor — the object we use to send SQL commands
        cursor = connect.cursor()
        # Sends our SELECT query to the database
        cursor.execute(query)
        # Fetches every result row into a list
        row = cursor.fetchall()
        # Closes the cursor after we're done with it
        cursor.close()
        # Closes the database connection to free up server resources
        connect.close()
        logger.info(f"Retrieved {len(row)} image records from database")
        return row
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return []

#             -------- READING A SINGLE IMAGE ---------

# Defines a function that loads one image file into a numpy array.
# Takes a single file path as input.
def readImage(path):
    # Grabs the logger again (safe to call multiple times — it reuses the same logger instance instead of creating duplicates).
    logger = utils.setupLogger()
    # cv2.imread() needs a string, not a Path object, so we convert it.
    # Returns a numpy array on success, or None if the file can't be read.
    img = cv.imread(str(path))
    # Checks if the read failed — this is our first corruption check, even before validation.py runs its own checks later.
    if img is None:
        # Logs a warning so we have a record of which files failed to load.
        logger.warning(f"Failed to read image: {path}")
        # Returns None so calling code can skip this image.
        return None
    # Returns the successfully loaded image as a numpy array.
    return img


#             -------- BATCH LOADING FROM DISK ---------

# Takes a list of paths (from load_from_disk) and actually reads each one into memory, pairing it with its original path.
def loadImagesBatch(imagePaths):
    # Logger again, for recording batch-level progress.
    logger = utils.setupLogger()
    # An empty list that will hold (path, image_array) tuples for every image that loads successfully.
    loaded = []
    # An empty list that will hold paths for every image that failed to load.
    failed = []
    # Loops through every path we were given.
    for path in imagePaths:
        # Calls our single-image function from above on this path.
        img = readImage(path)
        # Checks if the read succeeded.
        if img is not None:
            # Adds a tuple of (original path, loaded image array) to our results.
            loaded.append((path, img))
        else:
            # If it failed, we track the path separately so we know which files to flag or skip downstream
            failed.append(path)
    # Logs a summary of how the batch load went.
    logger.info(f"Loaded {len(loaded)} images successfully, {len(failed)} failed")
    # Returns both lists so the caller can decide what to do with each.
    return loaded, failed

# A generator function that yields one small batch at a time instead of loading everything into memory at once.
# chunkSize=50 means we process 50 images, free them, then move to the next 50.
def loadImagesInChunks(imagePaths, chunkSize=50):
    # range(0, total, chunkSize) steps through the list in chunkSize increments
    for i in range(0, len(imagePaths), chunkSize):
        # Slices just the current chunk of paths out of the full list
        chunk = imagePaths[i:i + chunkSize]
        # Loads only this chunk's worth of images into memory
        loaded, failed = loadImagesBatch(chunk)
        yield i, loaded, failed


#             -------- BATCH LOADING FROM SQL  ---------

# Defines a function for pulling image references from a SQL database.
# Left as a stub/placeholder until you have real DB credentials to test.
def loadFromSQL(dbRows):
    # Logger setup, same as the other functions.
    logger = utils.setupLogger()
    # Will hold (image_id, path, label, image_array) tuples for successful loads
    loaded = []
    # Will hold (image_id, path, label, image_array) tuples for failed loads
    failed = []
    # Loops through each database row
    for row in dbRows:
        # Unpacks the three columns from each row tuple
        imageID, imagePath, label = row
        # Converts the file path string from the DB into a Path object
        path = Path(imagePath)
        # Checks if the file actually exists at the path stored in the DB
        if not path.exists():
            # Logs a warning — the DB record exists but the file doesn't
            logger.warning(f"File not found on disk: {imagePath} (id={imageID})")
            # Tracks this as a failed load.
            failed.append((imageID, imagePath))
            continue
        # Attempts to load the image from disk using the DB-stored path.
        img = readImage(path)
        if img is not None:
            # Stores the full context: DB id, path, label, and the image array
            loaded.append((imageID, path, label, img))
        else:
            logger.warning(f"Failed to read image: {imagePath} (id={imageID})")
            failed.append((imageID, imagePath))
    # Logs a summary of the batch load from SQL records
    logger.info(f"SQL batch load: {len(loaded)} loaded successfully, {len(failed)} failed")
    # Returns both lists so the caller knows exactly what worked and what didn't
    return loaded, failed


#             -------- QUICK TEST BLOCK ---------

# Runs only when you execute "python ingestion.py" directly
if __name__ == "__main__":
    # Calls our disk-loading function using the default folder from config.py.
    paths = loadFromDisk()
    # Prints the count so you can see results immediately in the terminal.
    print(f"{len(paths)} image paths")
    # Checks if we actually found any images to test with.
    if paths:
        # Checks if we actually found any images to test with.
        loaded, failed = loadImagesBatch(paths[:5])
        # Tests loading just the first 5 images, instead of your whole dataset — keeps the test fast while still proving the logic works.
        print(f"Loaded: {len(loaded)}, Failed: {len(failed)}")
        if loaded:
            # Unpacks the first successful (path, image) tuple from our results.
            firstPath, firstImg = loaded[0]
            # Prints the array's dimensions (height, width, channels) — a quick way to confirm it's actually a valid loaded image.
            print(f"First image shape: {firstImg.shape}")

