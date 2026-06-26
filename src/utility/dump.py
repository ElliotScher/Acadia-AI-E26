#!/usr/bin/env python

import argparse
import datetime as dt
import glob
import os
import shutil
import sys


def dumpcamera(camdir: str, camname: str, destination: str):
    for child in os.scandir(camdir):
        if child.is_dir():
            dumpcamera(child.path, camname, destination)
        else:
            mtime = dt.datetime.fromtimestamp(child.stat().st_mtime)
            path = os.path.join(
                destination,
                camname,
                mtime.strftime("%Y-%m-%d"),
                mtime.strftime("%H-%M-%S.jpg"),
            )
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            elif not os.path.isdir(dirname):
                print(
                    "Destination exists and is not a directory: " + dirname,
                    file=sys.stderr,
                )
                exit(1)
            shutil.copy2(child.path, path)


def main():
    parser = argparse.ArgumentParser(
        prog="dump",
        description="Dump camera photos to a shared drive.",
    )
    parser.add_argument(
        "destination",
        nargs=1,
        action="store",
        help="The destination to dump to. This should be a directory in the shared drive named after a group, NOT one of the camera directories.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="The inputs to add to the destination. This sould be a list of directories which are named after the camera the images in it were obtained from. The directories are recursively searched for photos to add to the destination.",
    )
    args = parser.parse_args()

    # verify inputs are ok
    if not os.path.isdir(args.destination[0]):
        print("Destination must be a directory.", file=sys.stderr)
        exit(1)
    for input in args.inputs:
        dirs = glob.glob(input)
        if len(dirs) == 0:
            print("Input has no valid glob matches: " + input, file=sys.stderr)
            exit(1)
        for dir in dirs:
            if not os.path.isdir(dir):
                print(
                    "Input must be a camera output directory: " + input, file=sys.stderr
                )
                exit(1)

    # actually run the thing
    for input in args.inputs:
        for dir in glob.iglob(input):
            dumpcamera(dir, os.path.basename(os.path.abspath(dir)), args.destination[0])


if __name__ == "__main__":
    main()
