import math
from sqlalchemy.orm import Session
from db.models import Image, Instance


def merge_bikes_riders(session: Session, image: Image, threshold: float):
    instances = image.get_instances(session)
    bikes: list[Instance] = [x for x in instances if x.type_id == 1 or x.type_id == 3]
    people: list[Instance] = [x for x in instances if x .type_id == 0]
    merges: list[list[Instance]] = []
    merged: set[Instance] = set()

    for bike in bikes:
        btx = bike.x + bike.width / 2
        bty = bike.y
        merge = [bike]
        merges.append(merge)
        for person in people:
            ptx = person.x + person.width / 2
            pty = person.y + person.height / 2
            dist = math.sqrt((btx - ptx) ** 2 + (bty - pty) ** 2)
            if person not in merged and dist < person.width / 2 * threshold:
                merge.append(person)
                merged.add(person)

    for merge in merges:
        bike = merge.pop(0)
        for person in merge:
            x1 = min(bike.x, person.x)
            y1 = min(bike.y, person.y)
            x2 = max(bike.x + bike.width, person.x + person.width)
            y2 = max(bike.y + bike.height, person.y + person.height)
            bike.x = x1
            bike.y = y1
            bike.width = x2 - x1
            bike.height = y2 - y1
            session.add(bike)
            session.delete(person)

    session.commit()
