import argparse
from pipeline import runPipeline
from multiprocessing import freeze_support

# Wraps the script logic in a function — cleaner than having bare code directly under the __main__ check below.
def main():
    # Creates an argument parser with a short description shown when someone runs "python run_pipeline.py --help"
    parser = argparse.ArgumentParser(description="Run the image cleaning pipeline")
    # --source flag: accepts either "disk" or "sql"
    parser.add_argument(
        "--source",
        type=str,
        default="disk",
        choices=["disk", "sql"],
        help="Where to load images from: 'disk' or 'sql'"
    )
    # --input flag: overrides the default raw images folder for disk mode
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the input folder - disk mode only. Defaults to config.raw_dir"
    )
    # --query flag: lets you pass a custom SQL query from the terminal instead of editing config.py every time
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom SQL query to run - sql mode only. Defaults to config.dbImageQuery "
    )

    # Reads what was actually typed on the command line
    args = parser.parse_args()
    print(f"Running pipeline with source='{args.source}', input='{args.input}'")
    # Calls the main pipeline function with all three arguments
    result = runPipeline(source=args.source, input_dir=args.input)
    # Checks if the pipeline returned a result (it returns None if it stopped early)
    if result:
        cleaned, errors = result
        print(f"Done. {cleaned} images cleaned, {errors} errors.")
    
if __name__ == "__main__":
    freeze_support()
    main()