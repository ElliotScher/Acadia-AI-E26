import requests
import dotenv
import datetime
import time
import argparse
import sys
import os

dotenv.load_dotenv()

parser = argparse.ArgumentParser(
    prog="spypoint-dump",
    description="Dump camera photos from Spypoint to a shared drive.",
)
parser.add_argument("destination",
    nargs=1,
    action="store",
    help="The destination to dump to. This should be a directory in the shared drive named after a group, NOT one of the camera directories.")
parser.add_argument("camera",
    nargs=1,
    action="store",
    help="The ID of the camera to download photos from, or 'list' to list all cameras on the Spypoint account")
parser.add_argument("-s", "--size", help="Photo size to download: 'small' (72x72), 'medium' (160x90), or 'large' (720x406)", default="large", choices=("small", "medium", "large"))
parser.add_argument("-m", "--max", help="Maximum number of photos to download", type=int)

args = parser.parse_args()

if not os.path.isdir(args.destination[0]):
    print("Destination must be a directory.", file=sys.stderr)
    exit(1)

print("Logging in...")

login_request = requests.post("https://restapi.spypoint.com/api/v3/user/login", json={
    "username": os.getenv("SPYPOINT_USERNAME"),
    "password": os.getenv("SPYPOINT_PASSWORD")
})

login = login_request.json()
uuid = login["uuid"]
token = login["token"]

if args.camera[0] == "list":
    print("Logged in, listing cameras...")

    camera_list_request = requests.get("https://restapi.spypoint.com/api/v3/camera/base-info", headers={
        "Authorization": "bearer " + token 
    })

    camera_list = camera_list_request.json()

    print("Found %s cameras:" % (len(camera_list)))

    for camera in camera_list:
        print("- %s - %s (%s)" % (camera["name"], camera["model"], camera["id"]))
    exit(0)

camera_id = args.camera[0]
print("Logged in, downloading photos from camera %s..." % camera_id)

end_date = "2100-01-01T00:00:00.000Z"
total = 0
downloaded = 0

while end_date != "done":
    photo_list_request = requests.post("https://restapi.spypoint.com/api/v3/photo/all", json={
        "camera": [camera_id],
        "customTags": [],
        "dateEnd": end_date,
        "limit": 100,
        "mediaTypes": [],
        "species": [],
        "timeOfDay": []
    }, headers={
        "Authorization": "bearer " + token 
    })

    photo_list = photo_list_request.json()
    total += photo_list["countPhotos"]

    for photo in photo_list["photos"]:
        date = photo["originDate"][0:10]
        time = photo["originDate"][11:19].replace(":", "-")
        timestamp = datetime.datetime.strptime(photo["originDate"][0:19], "%Y-%m-%dT%H:%M:%S")
        path = os.path.join(
            args.destination[0],
            date,
            time + ".jpg"
        )

        try:
            os.mkdir(os.path.join(
                args.destination[0],
                date
            ))
        except FileExistsError:
            pass

        if os.path.exists(path):
            print("\nDownloaded %i photos" % (downloaded))
            exit(0)

        photo_request = requests.get("https://%s/%s" % (photo[args.size]["host"], photo[args.size]["path"]))

        with open(path, "wb") as file:
            file.write(photo_request.content)
            downloaded += 1
        os.utime(path, (timestamp.timestamp(), timestamp.timestamp()))
        
        print(f"\033[K{downloaded}/{str(total) if total % 100 != 0 else str(total) + "+"} photos...", end="\r")

        if args.max is not None and downloaded >= args.max:
            print("\nDownloaded %i photos" % (downloaded))
            exit(0)
    
    if photo_list["countPhotos"] < 100:
        end_date = "done"
    else:
        end_date = photo_list["photos"][photo_list["countPhotos"] - 1]["originDate"]

print("\nDownloaded %i photos" % (downloaded))