import requests
import datetime
import argparse
import sys
import os
from dataclasses import dataclass

uuid = ""
token = ""


@dataclass
class Photo:
    camera: str
    id: str
    date: datetime.datetime
    name: str
    size: int
    preview: list
    tag: list
    large: dict
    medium: dict
    small: dict

    def download(self, destination: str, size: str):
        if os.path.exists(destination):
            return
        rinfo = (
            self.small
            if size == "small"
            else (self.medium if size == "medium" else self.large)
        )
        dirname = os.path.dirname(destination)
        if not os.path.exists(dirname):
            os.makedirs(os.path.dirname(destination), exist_ok=True)
        elif not os.path.isdir(dirname):
            raise FileExistsError("Destination exists and is not a directory: " + dirname)
        with open(destination, "wb") as file:
            response = requests.get(f"https://{rinfo['host']}/{rinfo['path']}")
            file.write(response.content)


@dataclass
class Camera:
    id: str
    ucid: int
    installDate: str
    isCellular: bool
    name: str
    model: str

    def photos(
        self, limit: int = 100, before: datetime.datetime = datetime.datetime.now()
    ) -> list[Photo]:
        return [
            Photo(
                x["camera"],
                x["id"],
                datetime.datetime.fromisoformat(x["originDate"]),
                x["originName"],
                x["originSize"],
                x["preview"],
                x["tag"],
                x["large"],
                x["medium"],
                x["small"],
            )
            for x in requests.post(
                "https://restapi.spypoint.com/api/v3/photo/all",
                json={
                    "camera": [self.id],
                    "dateEnd": before.isoformat(),
                    "limit": limit,
                },
                headers={"Authorization": "bearer " + token},
            ).json()["photos"]
        ]

    def dump(
        self,
        destination: str,
        limit: int = 500,
        size: str = "large",
        before: datetime.datetime = datetime.datetime.now(),
    ):
        n = 0
        photos = self.photos(limit, before)
        while len(photos) > 0 and n < limit:
            for photo in photos:
                path = os.path.join(
                    destination,
                    self.name,
                    photo.date.strftime("%Y-%m-%d"),
                    photo.date.strftime("%H-%M-%S.jpg"),
                )
                photo.download(path, size)
                n += 1
            photos = self.photos(limit, photos[len(photos) - 1].date)


def setuser(id, tok):
    global uuid, token
    uuid = id
    token = tok


def login(username, password):
    global token, uuid
    response = requests.post(
        "https://restapi.spypoint.com/api/v3/user/login",
        json={
            "username": username,
            "password": password,
        },
    ).json()
    if "error" in response:
        raise PermissionError(f"Login error: {response['message']}.")
    uuid = response["uuid"]
    token = response["token"]


def listcams() -> list[Camera]:
    return [
        Camera(
            x["id"], x["ucid"], x["installDate"], x["isCellular"], x["name"], x["model"]
        )
        for x in requests.get(
            "https://restapi.spypoint.com/api/v3/camera/base-info",
            headers={"Authorization": "bearer " + token},
        ).json()
    ]


def main():
    parser = argparse.ArgumentParser(
        prog="spypoint-dump",
        description="Dump camera photos from Spypoint to a shared drive.",
    )
    parser.add_argument(
        "args",
        nargs="*",
        action="store",
        help="List of paths. The first is the destination directory, the following are treated as camera names.",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        default=False,
        help="List cameras before anything else.",
    )
    parser.add_argument(
        "-s",
        "--size",
        default="large",
        choices=("small", "medium", "large"),
        help="Photo size to download: 'small' (72x72), 'medium' (160x90), or 'large' (720x406)",
    )
    parser.add_argument(
        "-m",
        "--max",
        default=100,
        type=int,
        help="Maximum number of photos to download.",
    )
    parser.add_argument("-p", "--password", help="Login password.")
    parser.add_argument("-u", "--user", help="Login username.")
    args = parser.parse_args()

    # validate destination directory
    if len(args.args) > 0:
        destination = args.args[0]
        if not os.path.isdir(destination):
            print("Destination must be a directory.", file=sys.stderr)
            exit(1)

    # login and fetch cameras
    if len(args.args) > 0 or args.list:
        user = os.getenv("SPYPOINT_USERNAME") if not args.user else args.user
        passwd = (
            os.getenv("SPYPOINT_PASSWORD") if not args.password else args.password
        )
        if not user:
            print("Missing username.", file=sys.stderr)
        if not passwd:
            print("Missing password.", file=sys.stderr)
        if not user and not passwd:
            exit(1)
        try:
            login(user, passwd)
            cams = listcams()
        except Exception as e:
            print(e, file=sys.stderr)
            exit(1)

    # if we want to list cameras, then do that
    if args.list:
        for camera in cams:
            print(camera.name, camera.model, camera.id, sep="\t")

    # if we only wanted to list cameras, then we're done
    if len(args.args) < 1:
        exit(0)

    # ensure camera names specified actually exist
    namelist = args.args[1:] if len(args.args) > 1 else [x.name for x in cams]
    nameset = {x.name for x in cams}
    for camname in namelist:
        if camname not in nameset:
            print(f"Camera {camname} was not found on this account.", file=sys.stderr)
            exit(1)

    # dump cameras to destination
    print()
    for camera in [x for x in cams if x.name in namelist]:
        try:
            print(f"Downloading from {camera.name}...")
            camera.dump(destination, args.max, args.size)
        except Exception as e:
            print(e, file=sys.stderr)
            exit(1)


if __name__ == "__main__":
    main()
