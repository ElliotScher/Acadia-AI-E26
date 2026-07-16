from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db import get_db
from db.models import Image, Entity, Instance
import datetime as dt


def test_get_db_in_memory_creates_all_tables():
    engine = get_db(None)
    table_names = set(inspect(engine).get_table_names())
    assert {"image", "entity", "instance"}.issubset(table_names)


def test_get_db_file_backed_creates_file_on_disk(tmp_path):
    db_path = tmp_path / "test.sqlite"
    engine = get_db(str(db_path))
    assert db_path.exists()
    table_names = set(inspect(engine).get_table_names())
    assert {"image", "entity", "instance"}.issubset(table_names)


def test_foreign_keys_are_enforced_on_insert():
    engine = get_db(None)
    with Session(engine) as session:
        # entity_id 999 does not exist -> FK violation if pragma is respected
        orphan = Instance(
            image_id=1,
            entity_id=999,
            type_id=0,
            x=0,
            y=0,
            width=1,
            height=1,
            confidence=1.0,
        )
        session.add(orphan)
        try:
            session.commit()
            assert False, "expected IntegrityError due to dangling foreign key"
        except IntegrityError:
            session.rollback()


def test_foreign_keys_pragma_is_on_for_raw_connection():
    engine = get_db(None)
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA foreign_keys")).scalar()
        assert result == 1


def test_valid_instance_insert_succeeds_with_fk_enforced():
    engine = get_db(None)
    with Session(engine) as session:
        image = Image(path="/img.jpg", datetime=dt.datetime(2026, 1, 1, 0, 0, 0))
        entity = Entity(cluster=1)
        session.add_all([image, entity])
        session.flush()

        instance = Instance(
            image_id=image.id,
            entity_id=entity.id,
            type_id=0,
            x=0,
            y=0,
            width=1,
            height=1,
            confidence=1.0,
        )
        session.add(instance)
        session.commit()

        assert session.get(Instance, (image.id, entity.id)) is not None
