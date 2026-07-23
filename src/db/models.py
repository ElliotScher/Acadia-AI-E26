from __future__ import annotations

import csv
import datetime as dt
import os
from pathlib import Path

from sqlalchemy import (
    DDL,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Time,
    desc,
    event,
    exists,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    WriteOnlyMapped,
    mapped_column,
    relationship,
)

from detection.classes import CLASS_ID_MAPPING
from detection.direction.pedestrian_direction import process_single_image
from detection.image_yolo import DetectionResult
from utility.geometryutils import Rectangle


class Base(DeclarativeBase):
    pass


class Image(Base):
    __tablename__ = "image"
    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String(), unique=True, nullable=False)
    datetime: Mapped[dt.datetime] = mapped_column(DateTime(), nullable=False)
    analyzed: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)
    time: Mapped[dt.time] = mapped_column(Time(), nullable=False)

    instances: WriteOnlyMapped["Instance"] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
        single_parent=True,
        passive_deletes=True,
    )

    def __init__(self, **kwargs):
        super(Image, self).__init__(**kwargs)
        if (
            self.time is None
        ):  # editor may say code is unreachable, necessary for migrations
            self.time = dt.time(
                self.datetime.hour, self.datetime.minute, self.datetime.second
            )

    def get_instances(self, session: Session) -> list["Instance"]:
        return list(session.scalars(self.instances.select()).all())

    def get_entities(self, session: Session) -> list["Entity"]:
        return list(
            session.scalars(
                select(Entity).join(Instance).where(Instance.image_id == self.id)
            ).all()
        )

    def to_detection_result(self, session: Session) -> DetectionResult:
        return DetectionResult(
            Path(self.path).resolve(),
            list(map(lambda i: (
                Rectangle(i.x, i.y, i.width, i.height),
                i.entity_id,
                i.confidence
            ), self.get_instances(session)))
        )

    @staticmethod
    def import_from_dir(session: Session, dir: str):
        for root, _, files in os.walk(dir):
            for file in files:
                if not (
                    file.lower().endswith(".jpg")
                    or file.lower().endswith(".jpeg")
                    or file.lower().endswith(".png")
                ):
                    continue

                path = os.path.normpath(os.path.abspath(os.path.join(root, file)))
                if session.query(exists().where(Image.path == path)).scalar():
                    continue

                image = Image(
                    path=path,
                    datetime=dt.datetime.fromtimestamp(os.path.getmtime(path)),
                )
                session.add(image)
        session.commit()

    @staticmethod
    def get_earliest_image(session: Session) -> Image | None:
        try:
            return session.scalars(
                select(Image).order_by(Image.datetime).limit(1)
            ).one()
        except:
            return None

    @staticmethod
    def get_latest_image(session: Session) -> Image | None:
        try:
            return session.scalars(
                select(Image).order_by(desc(Image.datetime)).limit(1)
            ).one()
        except:
            return None

    @staticmethod
    def export_to_csv(
        session: Session, images: list[Image], path: str, interval: int = 0
    ):
        with open(path, mode="w", newline="") as file:
            writer = csv.writer(file)

            header = ["date", "time"]
            presentTypes = Instance.get_present_types(session)
            entityCounts: dict[int, int] = dict()
            clusterCount = 0
            for presentType in presentTypes:
                header.append(CLASS_ID_MAPPING[presentType] + " count")
                entityCounts[presentType] = 0
            header.append("cluster count")

            lastRangeTime: dt.datetime | None = None
            row: None | list[str | int] = None

            writer.writerows([header])
            data = []

            for image in images:
                rowTime: dt.datetime = image.datetime
                if interval != 0:
                    rowTime = rowTime - dt.timedelta(
                        minutes=rowTime.minute % interval,
                        seconds=rowTime.second,
                        microseconds=rowTime.microsecond,
                    )

                if lastRangeTime != rowTime:
                    if row is not None:
                        for entity in entityCounts.keys():
                            row.append(entityCounts[entity])
                            entityCounts[entity] = 0
                        row.append(clusterCount)
                        clusterCount = 0
                        data.append(row)

                        if len(data) > 100:
                            writer.writerows(data)
                            data = []

                    lastRangeTime = rowTime
                    row = [
                        rowTime.strftime("%Y-%m-%d"),
                        rowTime.strftime("%H:%M:%S"),
                    ]

                clusters = []
                for instance in image.get_instances(session):
                    entityCounts[instance.type_id] += 1
                    if instance.entity.cluster not in clusters:
                        clusters.append(instance.entity.cluster)
                clusterCount += len(clusters)

            if row is not None:
                for entity in entityCounts.keys():
                    row.append(entityCounts[entity])
                row.append(clusterCount)
                data.append(row)

            if len(data) > 0:
                writer.writerows(data)

    def __repr__(self) -> str:
        return f"Image({self.id})"


class Entity(Base):
    __tablename__ = "entity"
    id: Mapped[int] = mapped_column(primary_key=True)
    speed: Mapped[float] = mapped_column(Float(), nullable=True)
    ebike: Mapped[bool] = mapped_column(Boolean(), nullable=True)
    cluster: Mapped[int] = mapped_column(Integer(), nullable=True)

    instances: WriteOnlyMapped["Instance"] = relationship(
        back_populates="entity",
        cascade="all, delete-orphan",
        single_parent=True,
        passive_deletes=True,
    )

    def get_instances(self, session: Session) -> list["Instance"]:
        return list(session.scalars(self.instances.select()).all())

    def get_type_id(self, session: Session) -> int:
        return session.scalars(self.instances.select().limit(1)).one().type_id

    def get_earliest_image(self, session: Session) -> Image:
        return session.scalars(
            select(Image)
            .join(Instance)
            .where(Instance.entity_id == self.id)
            .order_by(Image.datetime)
            .limit(1)
        ).one()

    def get_latest_image(self, session: Session) -> Image:
        return session.scalars(
            select(Image)
            .join(Instance)
            .where(Instance.entity_id == self.id)
            .order_by(desc(Image.datetime))
            .limit(1)
        ).one()

    def get_timedelta(self, session: Session) -> dt.timedelta:
        images: list[Image] = list(
            session.scalars(
                select(Image)
                .join(Instance)
                .where(Instance.entity_id == self.id)
                .order_by(desc(Image.datetime))
            ).all()
        )
        if len(images) < 2:
            return dt.timedelta()
        return images[0].datetime - images[-1].datetime

    def get_entities_in_cluster(self, session: Session) -> list["Entity"]:
        return list(
            session.scalars(select(Entity).where(Entity.cluster == self.cluster)).all()
        )

    @staticmethod
    def export_to_csv(session: Session, entities: list[Entity], path: str):
        with open(path, mode="w", newline="") as file:
            writer = csv.writer(file)

            header = [
                "start date",
                "start time",
                "end date",
                "end time",
                "dwell time",
                "type",
                "cluster size",
            ]

            writer.writerows([header])
            data = []

            for entity in entities:
                earliestImage = entity.get_earliest_image(session)
                latestImage = entity.get_latest_image(session)
                row = [
                    earliestImage.datetime.strftime("%Y-%m-%d"),
                    earliestImage.datetime.strftime("%H:%M:%S"),
                    latestImage.datetime.strftime("%Y-%m-%d"),
                    latestImage.datetime.strftime("%H:%M:%S"),
                    (dt.datetime(1970, 1, 1) + entity.get_timedelta(session)).strftime(
                        "%H:%M:%S"
                    ),
                    CLASS_ID_MAPPING[entity.get_type_id(session)],
                    (
                        len(entity.get_entities_in_cluster(session))
                        if entity.cluster
                        else 0
                    ),
                ]

                data.append(row)

                if len(data) > 100:
                    writer.writerows(data)
                    data = []

            if len(data) > 0:
                writer.writerows(data)

    @staticmethod
    def export_clusters_to_csv(session: Session, entities: list[Entity], path: str):
        with open(path, mode="w", newline="") as file:
            writer = csv.writer(file)

            header = [
                "start date",
                "start time",
                "end date",
                "end time",
                "dwell time",
                "cluster size",
            ]

            writer.writerows([header])
            data = []
            clusters: list[int] = []

            for entity in entities:
                if entity.cluster in clusters:
                    continue

                clusters.append(entity.cluster)

                entities = entity.get_entities_in_cluster(session)
                earliestImage = entities[0].get_earliest_image(session)
                latestImage = entities[0].get_latest_image(session)

                for i in range(1, len(entities)):
                    entity = entities[i]

                    thisEarliestImage = entity.get_earliest_image(session)
                    if thisEarliestImage.datetime < earliestImage.datetime:
                        earliestImage = thisEarliestImage
                    thisLatestImage = entity.get_latest_image(session)
                    if thisLatestImage.datetime < latestImage.datetime:
                        latestImage = thisLatestImage

                row = [
                    earliestImage.datetime.strftime("%Y-%m-%d"),
                    earliestImage.datetime.strftime("%H:%M:%S"),
                    latestImage.datetime.strftime("%Y-%m-%d"),
                    latestImage.datetime.strftime("%H:%M:%S"),
                    (
                        dt.datetime(1970, 1, 1)
                        + (latestImage.datetime - earliestImage.datetime)
                    ).strftime("%H:%M:%S"),
                    len(entities),
                ]

                data.append(row)

                if len(data) > 100:
                    writer.writerows(data)
                    data = []

            if len(data) > 0:
                writer.writerows(data)

    def __repr__(self) -> str:
        return f"Entity({self.id})"


class Instance(Base):
    __tablename__ = "instance"
    image_id: Mapped[int] = mapped_column(
        ForeignKey(Image.id, ondelete="cascade", onupdate="restrict"), primary_key=True
    )
    entity_id: Mapped[int] = mapped_column(
        ForeignKey(Entity.id, ondelete="cascade", onupdate="no action"),
        primary_key=True,
    )
    type_id: Mapped[int] = mapped_column(Integer(), nullable=False)
    x: Mapped[int] = mapped_column(Integer(), nullable=False)
    y: Mapped[int] = mapped_column(Integer(), nullable=False)
    width: Mapped[int] = mapped_column(Integer(), nullable=False)
    height: Mapped[int] = mapped_column(Integer(), nullable=False)
    confidence: Mapped[float] = mapped_column(Float(), nullable=False)
    direction_lr: Mapped[int] = mapped_column(Integer(), nullable=True)
    direction_fb: Mapped[int] = mapped_column(Integer(), nullable=True)

    image: Mapped[Image] = relationship(back_populates="instances")
    entity: Mapped[Entity] = relationship(back_populates="instances")

    def overlap_with(self, other: Instance) -> float:
        ax1 = self.x
        ay1 = self.y
        ax2 = self.x + self.width
        ay2 = self.y + self.height
        bx1 = other.x
        by1 = other.y
        bx2 = other.x + other.width
        by2 = other.y + other.height

        intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0, min(ay2, by2) - max(ay1, by1)
        )
        union = (self.width * self.height) + (other.width * other.height) - intersection
        return intersection / union

    def __repr__(self) -> str:
        return f"Instance({self.image_id}, {self.entity_id})"

    @staticmethod
    def get_present_types(session: Session) -> list[int]:
        return list(session.scalars(select(Instance.type_id).distinct()))

    def analyze_pose_direction(self, session: Session, model, conf, minPoints):
        if self.type_id != 0 and self.type_id != 1:
            return

        directions = process_single_image(
            model,
            Path(self.image.path).resolve(),
            Path(),
            Path(),
            False,
            conf,
            [0],
            (
                self.x,
                self.y,
                self.x + self.width,
                self.y + self.height,
            ),
            minPoints,
        )

        if len(directions) > 0:
            self.direction_fb = directions[0].front_back
            self.direction_lr = directions[0].left_right
            session.add(self)
            session.commit()


trigger_ddl = DDL("""
CREATE TRIGGER IF NOT EXISTS delete_entity_when_last_instance_deleted
AFTER DELETE ON instance
FOR EACH ROW
WHEN NOT EXISTS (
  SELECT 1 FROM instance WHERE entity_id = OLD.entity_id
)
BEGIN
  DELETE FROM entity WHERE id = OLD.entity_id;
END;
""")

trigger2_ddl = DDL("""
CREATE TRIGGER IF NOT EXISTS delete_entity_when_last_instance_updated
AFTER UPDATE OF entity_id ON instance
FOR EACH ROW
WHEN NOT EXISTS (
  SELECT 1 FROM instance WHERE entity_id = OLD.entity_id
)
BEGIN
  DELETE FROM entity WHERE id = OLD.entity_id;
END;
""")

event.listen(
    Base.metadata,
    "after_create",
    trigger_ddl.execute_if(dialect="sqlite"),
)
event.listen(
    Base.metadata,
    "after_create",
    trigger2_ddl.execute_if(dialect="sqlite"),
)


@event.listens_for(Image.metadata, "after_create")
def add_column_if_not_exists(target, connection, **kwargs):
    try:
        connection.execute(DDL("""
SELECT analyzed FROM image LIMIT 1
"""))
    except:
        connection.execute(
            DDL("ALTER TABLE image ADD COLUMN analyzed BOOLEAN NOT NULL DEFAULT FALSE")
        )

    try:
        connection.execute(DDL("""
SELECT time FROM image LIMIT 1
"""))
    except:
        raise AssertionError(
            "Database is missing image time column. Delete database and rerun."
        )


@event.listens_for(Instance.metadata, "after_create")
def add_column_if_not_exists(target, connection, **kwargs):
    try:
        connection.execute(DDL("""
SELECT confidence FROM instance LIMIT 1
"""))
    except:
        connection.execute(DDL("ALTER TABLE instance ADD COLUMN confidence FLOAT"))
    try:
        connection.execute(DDL("""
SELECT direction_lr FROM instance LIMIT 1
"""))
    except:
        connection.execute(DDL("ALTER TABLE instance ADD COLUMN direction_lr INTEGER"))
        connection.execute(DDL("ALTER TABLE instance ADD COLUMN direction_fb INTEGER"))
