from dataclasses import dataclass
from db.models import Image, Entity, Instance
from sqlalchemy.orm import Session
import datetime as dt


@dataclass
class Track:
    start: dt.datetime
    lastSeen: dt.datetime
    instances: list[Instance]


def entity_iou_tracking(
    session: Session,
    images: list[Image],
    trackingDuration: dt.timedelta = dt.timedelta(seconds=1),
):
    tracks: dict[Entity, Track] = dict()

    for image in images:
        for entity, track in tracks.items():
            if (
                image.datetime - track.lastSeen < trackingDuration
                and len(track.instances) > 1
            ):
                entity.rawSpeed = (
                    abs(
                        track.instances[0].center()[0] - track.instances[-1].center()[0]
                    )
                    / (track.lastSeen - track.start).total_seconds()
                )

                for instance in track.instances:
                    instance.direction_lr = (
                        1
                        if track.instances[0].center()[0]
                        < track.instances[-1].center()[0]
                        else -1
                    )
        tracks = dict(
            filter(
                lambda track: image.datetime - track[1].lastSeen < trackingDuration,
                tracks.items(),
            )
        )

        for instance in image.get_instances(session):
            bestEntity: Entity | None = None
            bestIou: float = 0
            for entity, track in tracks.items():
                if (
                    track.instances[-1].type_id != instance.type_id
                    or track.lastSeen == image.datetime
                ):
                    continue

                iou = instance.overlap_with(track.instances[-1])
                if iou > 0 and iou > bestIou:
                    bestIou = iou
                    bestEntity = entity

            if bestEntity is not None:
                tracks[bestEntity].lastSeen = image.datetime
                tracks[bestEntity].instances.append(instance)

                if instance.entity.id != bestEntity.id:
                    session.delete(instance.entity)
                    instance.entity = bestEntity
                    session.add(instance)
            else:
                tracks[instance.entity] = Track(
                    image.datetime, image.datetime, [instance]
                )

    for entity, track in tracks.items():
        if len(track.instances) > 1:
            entity.rawSpeed = (
                abs(
                    track.instances[0].center()[0] - track.instances[-1].center()[0]
                )
                / (track.lastSeen - track.start).total_seconds()
            )
            
            for instance in track.instances:
                instance.direction_lr = (
                    1
                    if track.instances[0].center()[0] < track.instances[-1].center()[0]
                    else -1
                )

    session.commit()
