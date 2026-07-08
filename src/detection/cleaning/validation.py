# Library that generates a "perceptual hash" for an image — a short fingerprint that's similar for visually similar images, even if
# the file bytes themselves are different (e.g. recompressed copies).
import imagehash
# Pillow's Image module — imagehash needs a PIL Image object as input, not a numpy array, so we'll convert when needed.
from PIL import Image
import cv2 as cv
import numpy as np
import config
import utils

#             -------- CORRUPTION CHECK ---------

# Takes a numpy image array (already loaded by ingestion.py) and checks whether it looks broken in some way beyond a failed read.
def isCurrupt(image):
    # If ingestion already failed to load it, it's corrupt by definition.
    if image is None:
        return True
    # An empty array means something went wrong during loading.
    if image.size == 0:
        return True
    # Unpacks the first two values of the array's shape — height and width
    height, width = image.shape[:2]
    # Compares both dimensions against the minimum size set in config.py
    if height < config.min_image_detection or width < config.min_image_detection:
        return True
    # A near-zero value means almost every pixel is the same — a strong sign of an all-black, all-white, or solid-color image.
    if np.std(image) < 1:
        return True
    return False


#             -------- DUPLICATE DETECTION ---------

# Generates a perceptual hash for a single image array.
def computeHash(image):
    # converts the color order so the conversion below works correctly
    imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # Converts the numpy array into a PIL Image object, which is the format imagehash actually requires
    pilImage = Image.fromarray(imageRGB)
    # compute a "perceptual hash"
    return imagehash.phash(pilImage)

# Takes a list of (path, image) tuples and finds groups of duplicates.
def findDuplicates(imageList):
    # empty dictionary that will map each computed hash to the list of paths that share that hash
    hashes = {}
    # Will hold lists of paths that are duplicates of each other.
    duplicateGroup = []
    # Loops through every (path, image) pair we were given.
    for path, image in imageList:
        # Computes the perceptual hash for this specific image
        imgHash = computeHash(image)
        # A flag to track whether this image matched an existing hash group.
        matched = False
        # Loops through hashes we've already seen, to compare against.
        for existingHash in hashes:
            # imagehash supports subtraction between two hashes — the result is a "distance" number. Lower means more similar.
            # We compare it against our threshold from config.py.
            if imgHash - existingHash <= config.duplicateImage:
                # If similar enough, add this path to that existing group
                hashes[existingHash].append(path)
                matched = True
                break
        # Start a new group in the dictionary with just this one path.
        if not matched:
            # Start a new group in the dictionary with just this one path.
            hashes[imgHash] = [path]
    # Loops through all the groups we built up
    for group in hashes.values():
        # Only groups with more than one path actually contain duplicates.
        if len(group) > 1:
            duplicateGroup.append(group)
    # Returns a list of lists — each inner list is a group of duplicate paths
    return duplicateGroup


#             -------- FULL BATCH DETECTION ---------

# Takes the (path, image) tuples from ingestion.py and runs both corruption and 
# duplicate checks across the whole batch
def validateBatch(loadedImages):
    # Grabs our shared logger to record results
    logger = utils.setupLogger()
    # Creates a ManifestWriter so we can log every image's outcome.
    manifest = utils.ManifestWriter()
    # Will hold (path, image) tuples that pass all checks.
    valid = []
    # Will hold (path, image) tuples for anything that fails.
    rejected = []

    # Loops through every loaded image to check for corruption first.
    for path, image in loadedImages:
        if isCurrupt(image):
            # Records this path with the reason "corrupt".
            rejected.append((path, "corrupt"))
            # Writes this outcome into the manifest CSV
            manifest.log_entry(str(path), "disk", "validation", "rejected", "corrupt")
        else:
            # If it passed the corruption check, keep it for now
            valid.append((path, image))
    
    # If it passed the corruption check, keep it for now — duplicate checking happens next, separately.
    duplicateGroups = findDuplicates(valid)
    # collect every path that's part of any duplicate group.
    duplicatePaths = set()
    # Loops through each group of duplicates found
    for group in duplicateGroups:
        # Keeps the FIRST path in each group as the "original", and treats every path after it [1:] as a duplicate to remove.
        for dupPath in group[1:]:
            # Adds this duplicate path to our tracking set.
            duplicatePaths.add(dupPath)
    
    # Will hold the final list of images that are neither corrupt nor duplicates
    finalValid = []
    # Loops through everything that passed the corruption check
    for path, image, in valid:
        # Checks if this specific path was flagged as a duplicate.
        if path in duplicatePaths:
            rejected.append((path, "duplicate"))
            manifest.log_entry(str(path), "disk", "validation", "rejected", "duplicate")
        else:
            finalValid.append((path, image))
            manifest.log_entry(str(path), "disk", "validation", "passed", "")
    # Logs a summary of the whole batch validation run
    logger.info(f"Validation complete: {len(finalValid)} valid, {len(rejected)} rejected")
    # Returns the final valid images and the full rejected list with reasons.
    return finalValid, rejected


#             -------- QUICK TEST BLOCK ---------
    
if __name__ == "__main__":
    import ingestion
    paths = ingestion.loadFromDisk()
    print(f"Found {len(paths)} image paths")

    # tracks totals across all chunks
    totalValid = 0
    totalRejected = 0

    for chunkIndex, loaded, failed in ingestion.loadImagesInChunks(paths, chunkSize=50):
        valid, rejected = validateBatch(loaded)
        totalValid += len(valid)
        totalRejected += len(rejected)

        currentChunk = chunkIndex // 50 + 1
        totalChunks = (len(paths) + 49) // 50

        print(f"Chunk {currentChunk} / {totalChunks} ---------- {len(valid)} valid, {len(rejected)} rejected")
    print(f"\nFinal - Valid: {totalValid}, Rejected: {totalRejected}")




