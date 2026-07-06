from sqlalchemy import Select, select, func, or_, and_
from sqlalchemy.orm import Session
from db.models import Image, Instance


def merge_bikes_riders(session: Session, threshold: float):
    images: list[Image] = session.scalars(
        select(Image)
        .join(Instance, Image.id == Instance.image_id).where(
            or_(
                Instance.type_id == 1,  # bicycle
                Instance.type_id == 3,  # motorcycle
            )
        ).distinct()
    ).all()

    for image in images:
        instances = image.get_instances(session)
        bikes = filter(lambda i : i.type_id == 1 or i.type_id == 3, instances)
        people = filter(lambda i : i.type_id == 0, instances)

        for bike in bikes:
            for person in people:
                if person not in session.deleted and bike.overlap_with(person) >= threshold:
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
    
    session.commit()
