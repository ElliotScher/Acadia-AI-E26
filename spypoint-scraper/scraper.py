import requests
import dotenv
import os

dotenv.load_dotenv()

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

photo_list_request = requests.post("https://restapi.spypoint.com/api/v3/photo/all", json={
    "camera": [camera_id],
    "customTags": [],
    "dateEnd": "2100-01-01T00:00:00.000Z",
    "limit": 100,
    "mediaTypes": [],
    "species": [],
    "timeOfDay": []
}, headers={
    "Authorization": "bearer " + token 
})

photo_list = photo_list_request.json()

try:
    os.mkdir("spypoint-scraper/photos")
except FileExistsError:
    pass

downloaded = 0
for photo in photo_list["photos"]:
    if os.path.exists("spypoint-scraper/photos/%s.jpg" % (photo["id"])):
        continue

    photo_request = requests.get("https://%s/%s" % (photo["large"]["host"], photo["large"]["path"]))

    with open("spypoint-scraper/photos/%s.jpg" % (photo["id"]), "wb") as file:
        file.write(photo_request.content)
        downloaded += 1

print("Downloaded %i photos" % (downloaded))