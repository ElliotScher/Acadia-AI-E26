"""
Central place for all settings used across the pipeline.
Other files import from here instead of hardcoding values directly.
"""



"""
Path lets us build file paths that work correctly on both Windows and Mac/Linux, 
instad of manually typing \\ or / which breaks across operating system

os lets us read environment variables later (used for SQL credentials)
"""
from pathlib import Path
import os

#             -------- DIRECTORY PATH ---------

"""
__file__ is the path to this config.py file itself.
.resolve() converts it to a full absolute path (no "..").
.parent goes up one folder (from src/ to image_pipeline/).
.parent again would go up two folders, but here we only need src/'s parent, so this gives us the project root: image_pipeline/
"""
base_dir = Path(__file__).resolve().parent.parent
# Builds the path image_pipeline/data/raw using the / operator, which Path overloads to join folder names safely.
raw_dir = base_dir / "data" / "raw"
# Builds image_pipeline/data/cleaned — where pipeline output gets saved.
cleaned_dir = base_dir / "data" / "cleaned"
# Builds image_pipeline/logs — where the manifest CSV and log files live.
log_dir = base_dir / "logs"
# Full path to the manifest file itself, built from LOG_DIR.
manifest_dir = log_dir / "manifest.csv"
# Images that passed validation (not corrupt, not duplicate)
valid_dir = base_dir / "data" / "sorted" / "valid"
# Images that failed validation — corrupt or duplicate
invalid_dir = base_dir / "data" / "sorted" / "invalid"
# Images flagged as outliers BEFORE fixing — kept for reference.
outlier_dir = base_dir / "data" / "sorted" / "outlier"
# Outlier images AFTER being automatically repaired
fixed_dir = base_dir / "data" / "sorted" / "fixed"


#             -------- IMAGE VALIDATION SETTING ---------

# A set (not a list) of allowed file extensions.
# Using a set makes "is this extension allowed?" checks faster (O(1) lookup).
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
# Any image smaller than 20 pixels in width or height gets flagged as invalid; catches broken thumbnails or placeholder images.
min_image_detection = 20


#             -------- RESIZE SETTING ---------
# The width (in pixels) every image will be resized to before training.
targetWidth = 416
# Same idea, for height. Equal width/height here since we're using square inputs.
targetHeight = 416
# A string flag your resize.py will check "pad" means preserve aspect ratio and fill extra space, instead of stretching the image.
resizeMethod = "pad"


#             -------- DENOISING SETTING ---------
# Sets Non-Local Means as the default denoising method unless overridden.
defaultDenoise = "bilateral"
# The "h" parameter OpenCV's NLM function uses higher number means stronger denoising (but also more risk of blurring real detail).
nlmStrength = 10


#             -------- OUTLIER DETECTION SETTING ---------
# Images whose embedding distance from their cluster center exceeds this
# number of standard deviations get flagged as outliers.
outlierDistance = 1.5
# Mean pixel value below this = image is too dark --> brighten it
brightnessLow = 40
# Mean pixel value above this = image is too bright--> darken it
brightnessHigh = 215
# Laplacian variance below this = image is too blurry --> sharpen it
sharpness = 100.0


#             -------- DUPLICATE DETECTION ---------

# Max allowed difference between two perceptual hashes for images to be considered duplicates. Lower number = stricter matching.
duplicateImage = 5


#             -------- PARALEL PROCESSING SETTING ---------
# Uses all available CPU cores by default.
maxWorker = os.cpu_count()
# How many images to load into memory at once
chunkSize = 50


#             -------- SQL DATABASE ---------

# The type of database you're connecting to.
dbType = os.environ.get("IMG_DB_TYPE", "sqlserver")
# Reads the database host from an environment variable named IMG_DB_HOST.
# Returns None if it isn't set yet — safe placeholder until you configure it.
dbHost = os.environ.get("PUT THE DB NAME HERE")
# Same pattern — reads the database name from an environment variable.
dbName = os.environ.get("PUT THE DB NAME HERE")
# The port number — 1433 is the default for SQL Server
dbPort = os.environ.get("IMG_DB_PORT", "1433")
# Reads the database username.
dbUser = os.environ.get("PUT THE DB NAME HERE")
# Reads the database password.
dbPassword = os.environ.get("PUT THE DB NAME HERE")
# The name of the table that holds your image records.
dbImageTable = os.environ.get("IMG_DB_TABLE", "images")
# The default SQL query used to pull image records from the database
dbImageQuery = f"""
    SELECT image_id, file_path, label
    FROM {dbImageTable}
    WHERE label in ('person', 'vehicle', 'bicycle')
"""
