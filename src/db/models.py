from __future__ import annotations
from sqlalchemy import (
    String,
    DateTime,
    Float,
    Integer,
    Boolean,
    ForeignKey,
    DDL,
    select,
    desc,
    event,
    exists,
)
from sqlalchemy.orm import (
    Mapped,
    WriteOnlyMapped,
    Session,
    DeclarativeBase,
    mapped_column,
    relationship,
)
from datetime import datetime, timedelta
import os
from pathlib import Path

from detection.yolo import process_single_image


class Base(DeclarativeBase):
    pass


class Image(Base):
    __tablename__ = "image"
    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String(), unique=True, nullable=False)
    datetime: Mapped[datetime] = mapped_column(DateTime(), nullable=False)
    analyzed: Mapped[bool] = mapped_column(Boolean(), nullable=False, default=False)

    instances: WriteOnlyMapped["Instance"] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
        single_parent=True,
        passive_deletes=True,
    )

    def get_instances(self, session: Session) -> list["Instance"]:
        return session.scalars(self.instances.select()).all()

    def get_entities(self, session: Session) -> list["Instance"]:
        return session.scalars(
            select(Entity).join(Instance).where(Instance.image_id == self.id)
        ).all()

    def analyze(self, session: Session, model):
        if self.analyzed:
            for instance in self.get_instances(session):
                session.delete(instance)

        detections = process_single_image(
            model, Path(self.path).resolve(), Path(), Path(), False
        )

        for detection in detections:
            entity = Entity()
            instance = Instance(
                image=self,
                entity=entity,
                x=detection.box[0],
                y=detection.box[1],
                width=detection.box[2] - detection.box[0],
                height=detection.box[3] - detection.box[1],
                type_id=detection.cls_id,
                confidence=detection.conf,
            )
            session.add_all((entity, instance))

        self.analyzed = True
        session.add(self)
        session.commit()

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

                path = os.path.join(root, file)
                if session.query(exists().where(Image.path == path)).scalar():
                    continue

                image = Image(
                    path=path, datetime=datetime.fromtimestamp(os.path.getmtime(path))
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

    def __repr__(self) -> str:
        return f"Image({self.id})"


class Entity(Base):
    __tablename__ = "entity"
    id: Mapped[int] = mapped_column(primary_key=True)
    speed: Mapped[float] = mapped_column(Float(), nullable=True)
    direction: Mapped[int] = mapped_column(Integer(), nullable=True)
    ebike: Mapped[bool] = mapped_column(Boolean(), nullable=True)
    cluster: Mapped[int] = mapped_column(Integer(), nullable=True)

    instances: WriteOnlyMapped["Instance"] = relationship(
        back_populates="entity",
        cascade="all, delete-orphan",
        single_parent=True,
        passive_deletes=True,
    )

    def get_instances(self, session: Session) -> list["Instance"]:
        return session.scalars(self.instances.select()).all()

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

    def get_timedelta(self, session: Session) -> timedelta:
        images: List[Image] = session.scalars(
            select(Image)
            .join(Instance)
            .where(Instance.entity_id == self.id)
            .order_by(desc(Image.datetime))
        ).all()
        if len(images) < 2:
            return timedelta()
        return images[0].datetime - images[-1].datetime

    def get_entities_in_cluster(self, session: Session) -> list["Entity"]:
        return session.scalars(
            select(Entity).where(Entity.cluster == self.cluster)
        ).all()

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

    image: Mapped[Image] = relationship(back_populates="instances")
    entity: Mapped[Entity] = relationship(back_populates="instances")

    def __repr__(self) -> str:
        return f"Instance({self.image_id}, {self.entity_id})"


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


@event.listens_for(Instance.metadata, "after_create")
def add_column_if_not_exists(target, connection, **kwargs):
    try:
        connection.execute(DDL("""
SELECT confidence FROM instance LIMIT 1
"""))
    except:
        connection.execute(DDL("ALTER TABLE instance ADD COLUMN confidence FLOAT"))
