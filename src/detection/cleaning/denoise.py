import cv2 as cv
import config
import utils

#             -------- INDIVIDUAL DENOISING METHODS ---------

# Applies a Gaussian blur — fast, simple, good for mild general noise.
def gassian(image):
    # cv2.GaussianBlur needs a tuple for kernel size (width, height), and a sigma value
    return cv.GaussianBlur(image, (5, 5), 0)

# Applies a median filter — best for replaces each pixel with the median of its neighbors rather than averaging them.
def median(image):
    # takes image and a single odd kernel size value
    return cv.medianBlur(image, 5)

# Applies a bilateral filter — reduces noise while preserving edges, unlike Gaussian blur which softens edges along with noise
def bilateral(image):
    # takes image, diameter, sigma color, sigma space
    return cv.bilateralFilter(image, 9, 75, 75)

# Applies Non-Local Means denoising — generally the strongest and most effective method, but also the slowest of the four.
def nlm(image, h = None):
    # Uses the strength value passed in, or falls back to the default defined in config.py if none was given
    h_value = h if h is not None else config.nlmStrength
    # takes images, None(no input array provided), luminance, color component, template window size, search window size
    return cv.fastNlMeansDenoisingColored(image, None, h_value, h_value, 7, 21)

#             -------- DISPATCHER ---------

# A single entry point that picks the right denoising function based on a method name string, so calling code doesn't need
# to know which specific function to call.
def denoise(image, method=None):
    method = method or config.defaultDenoise
    logger = utils.setupLogger()
    # If method matches, log it and call that function.
    if method == "gaussian":
        logger.debug("Applying Gaussian Denoising")
        return gassian(image)
    elif method == "median":
        logger.debug("Applying Median Denoising")
        return median(image)
    elif method == "bilateral":
        logger.debug("Applying Bilateral Denoising")
        return bilateral(image)
    elif method == "nlm":
        logger.debug("Applying Non-Local Median Denoising")
        return nlm(image)
    else:
        logger.warning(f"Unknown denoise method '{method}', returning original image")
        return image

#             -------- QUICK TEST BLOCK ---------

if __name__ == "__main__":
    import ingestion
    paths = ingestion.loadFromDisk()
    if paths:
        # Loads just the first image. load_images_batch returns (loaded, failed) — we take [0] for loaded, then [0] again
        # for the first tuple, then unpack it into (_, img).
        _, img = ingestion.loadImagesBatch(paths[:1])[0][0]
        denoiseMethod = ["gaussian", "median", "bilateral", "nlm"]
        for method in denoiseMethod:
            # runs denoising with current method
            result = denoise(img, method=method)

            print(f"{method}: output shape {result.shape}")
            # Prints the method name and the resulting array shape, confirming each method ran without crashing and
            # returned an image of the same dimensions.



