import math
from sqlalchemy.orm import Session
from db.models import Image, Instance

    
def merge_bikes_riders(session: Session, image: Image, threshold: float):
    instances = image.get_instances(session)
    bikes: filter[Instance] = filter(
        lambda i: i.type_id == 1 or i.type_id == 3, instances
    )
    people: filter[Instance] = filter(lambda i: i.type_id == 0, instances)

    for bike in bikes:
        btx = bike.x + bike.width / 2
        bty = bike.y
        for person in people:
            ptx = person.x + person.width / 2
            pty = person.y + person.height / 2
            dist = math.sqrt((btx - ptx) ** 2 + (bty - pty) ** 2)
            if person not in session.deleted and (
                dist < person.width / 2 or bike.overlap_with(person) >= threshold
            ):
                x1 = min(bike.x, person.x)
                y1 = min(bike.y, person.y)
                x2 = max(bike.x + bike.width, person.x + person.width)
                y2 = max(bike.y + bike.height, person.y + person.height)
                width = x2 - x1
                height = y2 - y1

                session.delete(person)
                bike.x = x1
                bike.y = y1
                bike.width = width
                bike.height = height
                session.add(bike)
                break

    session.commit()
