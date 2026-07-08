import cv2 as cv
from pathlib import Path
import config
import utils
import denoise
import ingestion
import normalize
import outlierDetection
import resize
import validation
import numpy as np

# shutil.copy2() copies image files into the sorting folders while preserving the original file metadata (timestamps etc)
import shutil
# ThreadPoolExecutor runs multiple tasks simultaneously across threads.
# as_completed() lets us process results as each thread finishes, rather than waiting for all of them before moving on.
from concurrent.futures import ProcessPoolExecutor, as_completed


#             -------- FOLDER SETUP ---------

# Creates all output sorting folders at the start of a pipeline run
def createOutputFolder():
    for folder in [config.cleaned_dir, config.valid_dir, config.invalid_dir, config.valid_dir, config.outlier_dir , config.fixed_dir]:
        # Creates the folder and any missing parent folders.
        folder.mkdir(parents=True, exist_ok=True)


#             -------- SORT HELPER ---------

# Copies an image file into one of the sorting folders.
def sortImage(path, destinationFolder):
    sourcePath = Path(path).resolve()
    # Double-checks the file actually exists before trying to copy
    if not sourcePath.exists():
        raise FileNotFoundError(f"Source file not found: {sourcePath}")
    # Builds the destination path using just the filename
    dest = destinationFolder / Path(path).name
    # Copies the file — copy2 preserves original file timestamps
    shutil.copy2(str(sourcePath), str(dest))
    # Returns where it was copied to
    return dest


#             -------- SAVE HELPER ---------
# Takes a cleaned image array and writes it to the cleaned/ folder, reusing the original filename so it's traceable back to its source.
def savedCleanedImage(path, image):
    # Builds the output path using the same filename, but in the cleaned/ folder instead of raw/. path.name gets just the
    # filename (e.g. "image_001.jpg"), not the full original path.
    outputPath = config.cleaned_dir / Path(path).name
    # Checks if the image is still in float format (0-1 range) from the normalization step, rather than standard 0-255 integers
    if image.dtype != np.uint8:
        # Converts back to 0-255 range: multiply by 255, clip any values that went slightly out of bounds due to rounding,
        # then convert the data type back to uint8 for saving.
        imageToSave = (image * 255).clip(0, 255).astype(np.uint8)
    else:
        # If it's already uint8 (normalization wasn't applied, or only CLAHE ran without scaling), save it as-is.
        imageToSave = image
    # Writes the image file to disk. cv2.imwrite needs a string path, not a Path object, so we convert it.
    cv.imwrite(str(outputPath), imageToSave)    
    return outputPath


#             -------- WORKER FUNCTION ---------

# Takes a single tuple argument because ProcessPoolExecutor requires all arguments to be picklable — complex objects like ManifestWriter
# can't be passed directly, so we return log data instead of writing it.
def processImageWorker(args):
    # Unpacks the three values we need from the args tuple
    path, image, source = args
    try:
        # applies the configured denoising method
        denoised = denoise.denoise(image)
        # standardizes dimensions to targetWidth x targetHeight
        resized = resize.resize(denoised)
        # contrast normalization and pixel scaling
        normalized = normalize.normalize(resized)
        # Saves the fully processed image to the cleaned/ folder
        outputPath = savedCleanedImage(path, normalized)
        # Returns (success, path, source, output_path, error).The main process reads this and writes to the manifest.
        return True, str(path), source, str(outputPath), None
    except Exception as e:
        # Returns failure info — main process logs and counts it.
        return False, str(path), source, None, str(e)


#             -------- PARALLEL BATCH PROCESSING ---------
# Processes a chunk of images in parallel using multiple CPU threads.
# Returns counts of successes and failures for this chunk.
def processChunkParallel(inliers, source, batchManifest, logger):
    cleanedCount = 0
    errorCount = 0

    # Converts inliers to (path_string, image_array, source) tuples. path must be a string — Path objects aren't always picklable across
    # process boundaries on Windows
    argsList = [(str(path), image, source) for path, image in inliers]
    
    # Creates a thread pool with MAX_WORKERS threads running at once.
    # Each thread picks up one image and runs processSingleImage on it.
    with ProcessPoolExecutor(max_workers=config.maxWorker) as executor:
        # Submits all images in this chunk to the thread pool at once.
        futures = {
            executor.submit(processImageWorker, args): args[0]
            for args in argsList
        }
        # as_completed() yields each Future as it finishes — so we handle results as they come in, not in submission order
        for future in as_completed(futures):
            # Unpacks the (success, path, error) tuple from processSingleImage.
            success, path, source, outputPath, error = future.result()
            if success:
                # Adds to in-memory buffer — no file I/O yet
                batchManifest.addEntry(path, source, "pipeline", "completed", outputPath)
                cleanedCount += 1
            else:
                batchManifest.addEntry(path, source, "pipeline", "error", error)
                # Adds failure to buffer too
                logger.error(f"Failed: {path} - {error}")
                errorCount += 1
    # Writes ALL entries for this chunk in one single file operation.
    batchManifest.flush()
    return cleanedCount, errorCount



#             -------- MAIN PIPELINE FUNCTION ---------

# The main orchestrator function. source controls where images come from; input_dir lets you override the default raw folder.
def runPipeline(source="disk", input_dir=None, query=None):
    logger = utils.setupLogger()
    # Regular manifest for low-volume writes (validation, outlier stages)
    manifest = utils.ManifestWriter()

    # Batch manifest for high-volume writes (the cleaned/error entries from processing thousands of images through stages 4-6)
    batchManifest = utils.BatchManifestWriter()

    logger.info(f"Starting pipeline run — source: {source}, workers: {config.maxWorker}")

    # Creates all output folders before anything else runs
    createOutputFolder()

#             -------- STAGE 1: INGESTION ---------

    # Checks which source we're pulling from.
    if source == "disk":
        # Gets the list of valid file paths from the disk folder.
        paths = ingestion.loadFromDisk(input_dir)
        # Checks if ingestion found zero images at all.
        if not paths:
            logger.warning("No images found, stopping pipeline")
            return
        logger.info(f"Stage 1: {len(paths)} image paths found")

    elif source == "sql":
        # Fetches (image_id, file_path, label) rows from the database
        rows = ingestion.loadFromSQL(query)
        if not rows:
            logger.warning("No records returned from database, stopping pipeline")
            return
        logger.info(f"Stage 1 (Ingestion): {len(rows)} records from database")

    else:
        logger.error(f"Unsupported source: '{source}'. Use 'disk' or 'sql'.")
        return
    
#             -------- CHUNK PROCESSING ---------

    totalCleaned = 0
    totalErrors = 0
    totalRejected = 0
    totalOutliers = 0
    totalFixed = 0

    # calculates how many chucks we will process in total
    totalChunks = (len(paths) + config.chunkSize - 1) // config.chunkSize

    for chunkIndex, loaded, failed in ingestion.loadImagesInChunks(paths, chunkSize=config.chunkSize):
        currentChunk = chunkIndex // config.chunkSize + 1
        logger.info(f"--- Chunk {currentChunk} / {totalChunks} ({len(loaded)} images) ---")

        if not loaded:
            continue


#             -------- STAGE 2: VALIDATION ---------
    
    # Wraps this stage in a try/except so one stage's failure doesn't crash the entire pipeline run.
        try:
            valid, rejected = validation.validateBatch(loaded)
            for path, reason in rejected:
                try:
                    # path here comes directly from the loaded (path, image) tuple which still has its full Path object intact
                    sortImage(path, config.invalid_dir)
                except Exception as e:
                    logger.warning(f"  Could not sort invalid image {path}: {e}")
            totalRejected += len(rejected)
            logger.info(f" Validation: {len(valid)} valid, {len(rejected)} rejected")
        # Catches any unexpected error during this stage
        except Exception as e:
            logger.error(f"Validation failed on chunk {currentChunk}: {e}")
            return
        
        # Sort valid images too
        for path, _ in valid:
            try:
            # Copies valid images into data/sorted/valid/.
                sortImage(path, config.valid_dir)
            except Exception as e:
                logger.warning(f"  Could not sort valid image {path}: {e}")

#             -------- STAGE 3: OUTLIER DETECTION ---------
        try:
            # Flags statistical outliers using classical features
            inliers, outliers = outlierDetection.findOutliers(valid)
            # Logs immediately after detection so you can see if anything is being flagged at all — was missing before
            logger.info(
                f" Outlier Detection {len(inliers)} normal, "
                f"{len(outliers)} flagged in chunk {currentChunk}"            
            )
            # Will hold (path, fixed_image) tuples for repaired outliers
            fixedImages = []
            # Loops through every flagged outlier
            for path, image, distance in outliers:
                try:
                    # Resolve to absolute path before anything else — this is the key fix for the WinError 3 errors
                    fullPath = Path(path).resolve()
                    # Copies original outlier into data/sorted/outliers/ for reference
                    sortImage(fullPath, config.outlier_dir)
                    # Logs every single outlier saved so you can confirm the folder is actually being written to.
                    logger.info(f" Saved Outlier: {Path(path).name} (distance={distance:.2f})")
                    # Attempts to automatically repair the image
                    fixed, fixesApplied = outlierDetection.fixOutlier(image)
                    # If any fix was actually applied, save to fixed/ folder
                    if fixesApplied:
                        # Uses fullPath.name so the filename is consistent between the outlier/ and fixed/ folders
                        fixedPath = config.fixed_dir / fullPath.name
                        imageToSave = fixed
                        if fixed.dtype != np.uint8:
                            imageToSave = (fixed * 255).clip(0, 255).astype(np.uint8)
                        # Saves the repaired image to data/sorted/fixed/
                        success = cv.imwrite(str(fixedPath), imageToSave)
                        if success:
                            # Adds the fixed image to be processed through the remaining pipeline stages alongside the inliers
                            fixedImages.append((fullPath, fixed))
                            manifest.log_entry(str(path), source, "outlier_fix", "fixed", ", ".join(fixesApplied))
                            logger.info(
                                f" Fixed: {Path(path).name} - "
                                f"{', '.join(fixesApplied)}"
                            )
                            # Records exactly what was done to fix this image
                            totalFixed += 1
                        else:
                            # Catches the case where imwrite returns False (disk full, permissions issue, etc.)
                            logger.warning(f"  imwrite failed for: {fixedPath}")
                    else:
                        # If no fix could be applied, logs it as unfixable and excludes it from further processing.
                        manifest.log_entry(str(path), source, "outlier_fix", "unfixable", f"distance={distance:.2f}")
                        logger.info(
                            f"  Unfixable: {Path(path).name} "
                            f"(no applicable fix for distance={distance:.2f})"
                        )
                    
                except Exception as e:
                    logger.error(
                        f"  Error processing outlier {Path(path).name}: {e}"
                    )
                    continue

            totalOutliers += len(outliers)
            # Combines normal inliers AND successfully fixed outliers into one list to process through stages 4-6
            processQueue = inliers + fixedImages

        except Exception as e:
            logger.error(f"  Outlier detection failed on chunk {currentChunk}: {e}")
            processQueue = valid


#             -------- STAGES 4-6: DENOISE, RESIZE, NORMALIZE (per image) ---------

        cleaned, errors = processChunkParallel(processQueue, source, batchManifest, logger)
        totalCleaned += cleaned
        totalErrors += errors
        logger.info(f"  Processed: {cleaned} cleaned, {errors} errors")

#             -------- FINAL SUMMARY ---------
    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Total images found:    {len(paths)}")
    logger.info(f"  Rejected (invalid):    {totalRejected}")
    logger.info(f"  Outliers detected:     {totalOutliers}")
    logger.info(f"  Outliers fixed:        {totalFixed}")
    logger.info(f"  Successfully cleaned:  {totalCleaned}")
    logger.info(f"  Errors:                {totalErrors}")
    logger.info(f"  Output folders:")
    logger.info(f"    Valid:    {config.valid_dir}")
    logger.info(f"    Invalid:  {config.invalid_dir}")
    logger.info(f"    Outliers: {config.outlier_dir}")
    logger.info(f"    Fixed:    {config.fixed_dir}")
    logger.info(f"    Cleaned:  {config.cleaned_dir}")
    logger.info("=" * 50)
    
    # Returns the final counts so calling code (or tests) can check results
    return totalCleaned, totalErrors
        
