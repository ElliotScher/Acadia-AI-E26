import requests
import dotenv
import os

dotenv.load_dotenv()

PHOTO_SIZE = "medium" # small (72x72), medium (160x90), or large (720x406)

print("Logging in...")

login_request = requests.post("https://restapi.spypoint.com/api/v3/user/login", json={
    "username": os.getenv("SPYPOINT_USERNAME"),
    "password": os.getenv("SPYPOINT_PASSWORD")
})

login = login_request.json()
uuid = login["uuid"]
token = login["token"]

print("Logged in, listing cameras...")

camera_list_request = requests.get("https://restapi.spypoint.com/api/v3/camera/base-info", headers={
    "Authorization": "bearer " + token 
})

camera_list = camera_list_request.json()

print("Found %s cameras:" % (len(camera_list)))

camera_map = {}
for camera in camera_list:
    print("- %s - %s (%s)" % (camera["name"], camera["model"], camera["id"]))
    camera_map[camera["id"]] = camera["id"]
    camera_map[camera["name"]] = camera["id"]

selected_camera = None
while selected_camera not in camera_map:
    if selected_camera is not None:
        print("Camera not recognized, try again")
    selected_camera = input("Enter a camera name or ID to look up: ")

camera_id = camera_map[selected_camera]
print("Downloading photos from camera %s..." % camera_id)

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

    try:
        os.mkdir("spypoint-scraper/photos")
    except FileExistsError:
        pass

    for photo in photo_list["photos"]:
        date = photo["originDate"][0:10]
        time = photo["originDate"][11:19].replace(":", "-")
        try:
            os.mkdir("spypoint-scraper/photos/%s" % date)
        except FileExistsError:
            pass

        if os.path.exists(f"spypoint-scraper/photos/{date}/{time}.jpg"):
            continue

        photo_request = requests.get("https://%s/%s" % (photo[PHOTO_SIZE]["host"], photo[PHOTO_SIZE]["path"]))

        with open(f"spypoint-scraper/photos/{date}/{time}.jpg", "wb") as file:
            file.write(photo_request.content)
            downloaded += 1
        
        print(f"\033[K{downloaded}/{str(total) if total % 100 != 0 else str(total) + "+"} photos...", end="\r")
    
    if photo_list["countPhotos"] < 100:
        end_date = "done"
    else:
        end_date = photo_list["photos"][photo_list["countPhotos"] - 1]["originDate"]

print("\nDownloaded %i photos" % (downloaded))