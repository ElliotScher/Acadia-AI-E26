import cv2 as cv
import numpy as np
import config
import utils


#             -------- FEATURE EXTRACTION ---------

# Computes a small set of handcrafted statistics that describe an image numerically
def extractFeatures(image):
    # Converts the image to grayscale — most of these features only need brightness information, not full color.
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Average pixel intensity across the whole image.
    brightness = float(np.mean(gray))
    # Standard deviation of pixel intensity — low values mean a flat, washed-out image; high values mean strong contrast.
    contrast = float(np.std(gray))
    # Applies the Laplacian operator, which highlights areas of rapid intensity change (edges). CV_64F gives precise decimal output.
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    # Sharp, in-focus images have high variance; blurry images have low variance because there are fewer strong edges.
    sharpness = float(np.var(laplacian))
    # Detects edges using the Canny algorithm. 100 and 200 are the lower/upper thresholds controlling
    # how strong a gradient must be to count as an edge.
    edges = cv.Canny(gray, 100, 200)
    # Calculates what fraction of pixels are part of a detected edge. edges > 0 creates a True/False array
    edgeDensity = float(np.mean(edges > 0))
    # Unpacks height and width from the image's shape.
    height, width = image.shape[:2]
    # Width-to-height ratio — flags images with unusual proportions.
    aspectRatio = float(width / height)
    return np.array([brightness, contrast, sharpness, edgeDensity, aspectRatio])


#             -------- OUTLIER FIXER ---------

# Attempts to automatically repair a flagged outlier image.
# Runs three checks in sequence — dark, bright, blurry — and applies the appropriate correction for each one found
def fixOutlier(image):
    # Convert to grayscale to measure brightness and sharpness
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Measure current average brightness
    brightness = float(np.mean(gray))
    # Measure current sharpness
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    sharpness = float(np.var(laplacian))
    # Work on a copy so the original is never modified in place
    fixed = image.copy()
    # Tracks which fixes were applied — logged to the manifest
    fixedApplied = []

    if brightness < config.brightnessLow:
        # Convert to LAB so we can boost only the brightness channel (L)
        lab = cv.cvtColor(fixed, cv.COLOR_BGR2LAB)
        # Split into the three channels
        l, a, b = cv.split(lab)
        # Apply CLAHE to the L channel — lifts detail out of dark areas more naturally than just adding a flat brightness offset
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        # Merge channels back and convert to BGR
        fixed = cv.cvtColor(cv.merge([l, a, b]), cv.COLOR_LAB2BGR)
        # Records what was done
        fixedApplied.append("brightened")

    elif brightness > config.brightnessHigh:
        lab = cv.cvtColor(fixed, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        # Subtracts 40 from every L pixel value, darkening the image
        l = cv.subtract(l, 40)
        fixed = cv.cvtColor(cv.merge([l, a, b]), cv.COLOR_LAB2BGR)
        fixedApplied.append("darkened")

    # An unsharp masking kernel — the center weight (5) boosts the pixel, while the neighbors (-1) subtract the surrounding blur.
    # This is a fast, classical sharpening approach with no extra libraries.
    if sharpness < config.sharpness:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        # Applies the sharpening kernel across the whole image.
        # -1 means the output depth matches the input depth automatically.
        fixed = cv.filter2D(fixed, -1, kernel)
        fixedApplied.append("sharpened")
    return fixed, fixedApplied


#             -------- BATCH OUTLIER DETECTION ---------

# Takes a list of (path, image) tuples, computes features for all of them, and flags any that are statistical outliers.
def findOutliers(loadedImage):
    logger = utils.setupLogger()
    manifest = utils.ManifestWriter()
    if not loadedImage:
        return [], []
    # Computes the 5-feature vector for every image, then stacks them all into a single 2D array
    featureMatrix = np.array([extractFeatures(img) for _, img in loadedImage])
    # Calculates the average value of each feature ACROSS all images.
    meanVector = np.mean(featureMatrix, axis=0)
    # Calculates the standard deviation of each feature across all images
    stdVector = np.std(featureMatrix, axis=0)
    # Safety check: if any feature has zero variation (every image identical on that feature), replace 0 with a tiny number to
    # avoid a division-by-zero error in the next step.
    stdVector[stdVector == 0] = 1e-6
    # Converts every raw feature value into a z-score — how many standard deviations it is from the average.
    zScores = (featureMatrix - meanVector) / stdVector
    # Combines all 5 z-scores per image into a single distance number, using the standard Euclidean distance formula.   
    distances = np.sqrt(np.sum(zScores ** 2, axis=1))
    # Creates a True/False array: True wherever an image's combined distance exceeds the threshold set in config.py
    outlierMask = distances > config.outlierDistance
    # Will hold (path, distance) tuples for every flagged image.
    outliers = []
    # Will hold (path, image) tuples for every normal image.
    inliers = []
    # Loops through every image, keeping track of its index (i) so we can look up the matching distance and mask value.
    for i, (path, image) in enumerate(loadedImage):
        # Checks if this specific image was flagged as an outlier
        if outlierMask[i]:
            # Records the path along with how far out it scored
            outliers.append((path, image, float(distances[i])))
            # Logs this outcome to the manifest, including the score so you can review later why it was flagged
            manifest.log_entry(str(path), "disk", "outlier_detection", "flagged", f"distance = {distances[i]:.2f}")
        else:
            # If not flagged, it passes through to the next pipeline stage
            inliers.append((path, image))
    logger.info(f"Outlier detection complete: {len(inliers)} normal, {len(outliers)} flagged")
    return inliers, outliers

#             -------- QUICK TEST BLOCK ---------

if __name__ == "__main__":
    import ingestion
    paths = ingestion.loadFromDisk()
    if paths:
        totalNormal = 0
        totalFlagged = 0 
        totalChunks = (len(paths) + config.chunkSize - 1) // config.chunkSize

        for chunkIndex, loaded, failed in ingestion.loadImagesInChunks(paths, chunkSize=config.chunkSize):
            if not loaded:
                continue

            # Runs outlier detection on the whole batch
            inliers, outliers = findOutliers(loaded)
            totalNormal += len(inliers)
            totalFlagged += len(outliers)

            currentChunk = chunkIndex // config.chunkSize + 1
            print(f"Chunk {currentChunk}/{totalChunks} - "
                  f"Normal: {len(inliers)}, Flagged: {len(outliers)}")
            
            if outliers and currentChunk == 1:
                path, image, distance = outliers[0]
                fixed, fixes = fixOutlier(image)
                print(f"Fixed '{path.name}, Flagged: {totalFlagged}")
        print(f"\nFinal - Normal: {totalNormal}, Flagged: {totalFlagged}")

