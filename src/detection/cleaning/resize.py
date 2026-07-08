import cv2 as cv
import numpy as np
import config
import utils

#             -------- PADDING-BASED RESIZE ---------

# Resizes an image to fit within target dimensions while preserving its original aspect ratio, then pads the leftover space with black.
def resizeWithPadding(image, targetWidth = None, targetHeight = None):
    targetWidth = targetWidth or config.targetWidth
    targetHeight = targetHeight or config.targetHeight
    # Unpacks the image's current height and width from its shape.
    h, w = image.shape[:2]
    # Calculates a single scale factor that fits the image within the target box without distorting it.
    scale = min(targetWidth/w, targetHeight/h)
    # Calculates the new width after scaling, rounded to a whole pixel.
    newWidth = int(w * scale)
    # Calculates the new height after scaling, rounded to a whole pixel.
    newHeight = int(h * scale)
    # Resizes the image to the new dimensions.
    resized = cv.resize(image, (newWidth, newHeight), interpolation=cv.INTER_AREA)
    # Creates a blank black canvas of exactly the target size.
    canvas = np.zeros((targetHeight, targetWidth, 3), dtype='uint8')
    # Calculates how much empty space to leave above the resized image, centering it vertically
    padTop = (targetHeight - newHeight) // 2
    # same thing, but for horizontal centering
    padLeft = (targetWidth - newWidth) // 2
    # Pastes the resized image into the center of the blank canvas
    canvas[padTop:padTop + newHeight, padLeft:padLeft + newWidth] = resized
    return canvas


#             -------- STRETCH-BASED RESIZE ---------

# Resizes an image directly to the target dimensions without preserving aspect ratio.
def resizeStretch(image, targetWidth = None, targetHeight = None):
    targetWidth = targetWidth or config.targetWidth
    targetHeight = targetHeight or config.targetHeight
    # Directly resizes to the exact target dimensions, ignoring the original aspect ratio entirely
    return cv.resize(image, (targetWidth, targetHeight), interpolation=cv.INTER_CUBIC)


#             -------- DISPATCHER ---------

# A single entry point that picks the right resize function based on a method name, matching the pattern used in denoise.py.
def resize(image, method=None, targetWidth = None, targetHeight = None):
    method = method or config.resizeMethod
    logger = utils.setupLogger()
    if method == "pad":
        logger.debug("Resizing with padding (aspect ratio preserved)")
        return resizeWithPadding(image, targetWidth, targetHeight)
    elif method == "stretch":
        logger.debug("Resizing with stretch (aspect ratio distorted)")
        return resizeStretch(image, targetWidth, targetHeight)
    else:
        logger.warning(f"Unknown resize method '{method}', defaulting to padding")
        return resizeWithPadding(image, targetWidth, targetHeight)

#             -------- QUICK TEST BLOCK ---------

if __name__ == "__main__":
    import ingestion
    paths = ingestion.loadFromDisk()
    if paths:
        # Loads just the first image. load_images_batch returns (loaded, failed) — we take [0] for loaded, then [0] again
        # for the first tuple, then unpack it into (_, img).
        _, img = ingestion.loadImagesBatch(paths[:1])[0][0]
        print(f"Original shape: {img.shape}")
        padding = resize(img, method="pad")
        print(f"Padding shape: {padding.shape}")
        stretched = resize(img, method="stretch")
        print(f"Stretching shape: {stretched.shape}")

