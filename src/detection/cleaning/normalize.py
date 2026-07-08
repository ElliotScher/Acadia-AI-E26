import cv2 as cv
import numpy as np
import utils


#             -------- PIXL SCALE ---------

# Converts pixel values from the standard 0-255 integer range down to a 0-1 floating point range
def scalePixel(image):
    # .astype(np.float32) converts the array's data type from uint8 (whole numbers 0-255) to float32 (decimal numbers)
    return image.astype(np.float32) / 255.0


#             -------- CONTRAST NORMALIZATION ---------

# Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) — improves contrast in images with uneven lighting, without
# over-brightening already well-lit areas like simple histogram equalization tends to do.
def clahe(image):
    # Converts the image from BGR color space into LAB color space.
    # LAB separates brightness (L) from color information (A and B), so we can adjust contrast without distorting colors.
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    # Splits the LAB image into its three separate channels.
    l_channel, a_channel, b_channel = cv.split(lab)
    # create a CLAHE object
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Applies CLAHE only to the L (brightness) channel — this is why we converted to LAB instead of just doing this in BGR
    l_equalized = clahe.apply(l_channel)
    # Merges the equalized brightness channel back together with the original, untouched color channels.
    labEqualized = cv.merge((l_equalized, a_channel, b_channel))
    # Converts the image back from LAB into standard BGR format before returning it, so it matches every other image in the pipeline.
    return cv.cvtColor(labEqualized, cv.COLOR_LAB2BGR)


#             -------- DISPATCHER ---------

# A single entry point that applies one or more normalization steps in sequence, based on a list of method names.
def normalize(image, methods=None):
    # checks if no specific list was provided
    if methods is None:
        # Default order: fix contrast first, THEN scale pixel values.
        methods = ["clahe", "scale"]
    logger = utils.setupLogger()
    # Starts a working copy reference that we'll update after each step.
    result = image
    # Loops through each requested method in the order given.
    for method in methods:
        if method == "clahe":
            logger.debug("Applying CLAHE contrast normalization")
            # Runs CLAHE and updates result with the output.
            result = clahe(result)
        elif method == "scale":
            logger.debug("Applying pixel scaling (0-255 to 0-1)")
            # Runs pixel scaling and updates result with the output.
            result = scalePixel(result)
        else:
            logger.warning(f"Unknown normalize method '{method}', skipping")
    return result


#             -------- QUICK TEST BLOCK ---------

if __name__ == "__main__":
    import ingestion
    paths = ingestion.loadFromDisk()
    if paths:
        _, img = ingestion.loadImagesBatch(paths[:1])[0][0]
        # Prints the original data type (should be uint8) and the actual min/max pixel values found in this specific image.
        print(f"Original dtype: {img.dtype}, range: {img.min()} - {img.max()}")
        # Runs the full default normalization sequence (CLAHE then scaling).
        result = normalize(img)
        # Prints the new data type (should be float32) and the new min/max values, which should now fall between 0 and 1.
        print(f"Normalized dtype: {result.dtype}, range: {result.min()} - {result.max()}")





