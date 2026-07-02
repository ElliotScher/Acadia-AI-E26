from pytest import fixture
from db import get_db
from sqlalchemy.orm import Session
from sqlalchemy import exists, and_
from db.models import Image, Entity, Instance
from datetime import datetime


@fixture()
def test_session():
    global image1, image2, image3, image4, entity1, entity2, entity3, entity4, entity5, instance1, instance2, instance3, instance4, instance5, instance6, instance7, instance8, instance7

    # create in-memory database for testing
    db = get_db(None)

    # create testing data
    image1 = Image(
        path="/img1.jpg", datetime=datetime.fromisoformat("2026-06-26T12:14:02Z")
    )
    image2 = Image(
        path="/img2.jpg", datetime=datetime.fromisoformat("2026-06-26T12:14:04Z")
    )
    image3 = Image(
        path="/img3.jpg", datetime=datetime.fromisoformat("2026-06-26T12:15:37Z")
    )
    image4 = Image(
        path="/img4.jpg", datetime=datetime.fromisoformat("2026-06-26T12:15:38Z")
    )

    entity1 = Entity(cluster=1)
    entity2 = Entity(cluster=1)
    entity3 = Entity(cluster=2)
    entity4 = Entity(cluster=2)
    entity5 = Entity(cluster=3)

    instance1 = Instance(
        image=image1,
        entity=entity1,
        type_id=0,
        x=0,
        y=0,
        width=0,
        height=0,
        confidence=1,
    )
    instance2 = Instance(
        image=image1,
        entity=entity2,
        type_id=0,
        x=0,
        y=0,
        width=0,
        height=0,
        confidence=1,
    )
    instance3 = Instance(
        image=image2,
        entity=entity1,
        type_id=0,
        x=0,
        y=0,
        width=0,
        height=0,
        confidence=1,
    )
    instance4 = Instance(
        image=image2,
        entity=entity2,
        type_id=0,
        x=0,
        y=0,
        width=0,
        height=0,
        confidence=1,
    )
    instance5 = Instance(
        image=image3,
        entity=entity3,
        type_id=1,
        x=0,
        y=0,
        width=0,
        height=0,
        confidence=1,
    )
    instance6 = Instance(
        image=image3,
        entity=entity4,
        type_id=1,
        x=0,
        y=0,
        width=0,
        height=0,
        confidence=1,
    )
    instance7 = Instance(
        image=image4,
        entity=entity5,
        type_id=0,
        x=0,
        y=0,
        width=0,
        height=0,
        confidence=1,
    )
    instance8 = Instance(
        image=image4,
        entity=entity3,
        type_id=1,
        x=0,
        y=0,
        width=0,
        height=0,
        confidence=1,
    )
    instance9 = Instance(
        image=image4,
        entity=entity4,
        type_id=1,
        x=0,
        y=0,
        width=0,
        height=0,
        confidence=1,
    )

    # add testing data
    with Session(db) as session:
        session.add_all((image1, image2, image3, image4))
        session.add_all((entity1, entity2, entity3, entity4, entity5))
        session.add_all(
            (
                instance1,
                instance2,
                instance3,
                instance4,
                instance5,
                instance6,
                instance7,
                instance8,
                instance9,
            )
        )
        session.commit()

        yield session


def test_image(test_session: Session):
    instances: List[Instance] = image1.get_instances(test_session)
    assert len(instances) == 2
    assert instances[0].entity_id == entity1.id
    assert instances[1].entity_id == entity2.id

    entities: List[Entity] = image1.get_entities(test_session)
    assert len(entities) == 2
    assert entities[0].id == entity1.id
    assert entities[1].id == entity2.id

    assert Image.get_earliest_image(test_session).id == image1.id
    assert Image.get_latest_image(test_session).id == image4.id
    test_session.delete(image1)
    test_session.delete(image2)
    test_session.delete(image3)
    test_session.delete(image4)
    assert Image.get_earliest_image(test_session) is None
    assert Image.get_latest_image(test_session) is None


def test_entity(test_session: Session):
    instances: List[Instance] = entity1.get_instances(test_session)
    assert len(instances) == 2
    assert instances[0].image_id == image1.id
    assert instances[1].image_id == image2.id

    assert entity1.get_type_id(test_session) == 0
    assert entity3.get_type_id(test_session) == 1

    assert entity1.get_earliest_image(test_session).id == image1.id
    assert entity1.get_latest_image(test_session).id == image2.id
    assert entity1.get_timedelta(test_session).total_seconds() == 2
    assert entity5.get_timedelta(test_session).total_seconds() == 0

    entities_in_cluster: List[Entity] = entity1.get_entities_in_cluster(test_session)
    assert len(entities_in_cluster) == 2
    assert entities_in_cluster[0].id == entity1.id
    assert entities_in_cluster[1].id == entity2.id


def test_instance(test_session: Session):
    assert instance1.entity.id == entity1.id
    assert instance1.image.id == image1.id


def test_delete_entity(test_session: Session):
    test_session.delete(entity1)
    test_session.commit()

    assert not test_session.query(exists().where(Entity.id == entity1.id)).scalar()
    assert not test_session.query(
        exists().where(Instance.entity_id == entity1.id)
    ).scalar()


def test_delete_instance(test_session: Session):
    test_session.delete(instance1)
    test_session.commit()

    assert not test_session.query(
        exists().where(
            and_(
                Instance.entity_id == instance1.entity_id,
                Instance.image_id == instance1.image_id,
            )
        )
    ).scalar()

    entity1_id = entity1.id

    test_session.delete(instance3)
    test_session.commit()

    assert not test_session.query(
        exists().where(
            and_(
                Instance.entity_id == instance3.entity_id,
                Instance.image_id == instance3.image_id,
            )
        )
    ).scalar()
    assert not test_session.query(
        exists().where(Instance.entity_id == entity1_id)
    ).scalar()
    assert not test_session.query(exists().where(Entity.id == entity1_id)).scalar()


def test_delete_image(test_session: Session):
    test_session.delete(image1)
    test_session.commit()

    assert not test_session.query(exists().where(Image.id == image1.id)).scalar()
    assert not test_session.query(
        exists().where(Instance.image_id == image1.id)
    ).scalar()

    test_session.delete(image2)
    test_session.commit()

    assert not test_session.query(exists().where(Image.id == image2.id)).scalar()
    assert not test_session.query(
        exists().where(Instance.image_id == image2.id)
    ).scalar()
    assert not test_session.query(exists().where(Entity.cluster == 1)).scalar()


def test_create(test_session: Session):
    image5 = Image(
        path="/img5.jpg", datetime=datetime.fromisoformat("2026-06-26T12:16:27Z")
    )

    test_session.add(image5)
    test_session.commit()

    assert test_session.query(exists().where(Image.id == image5.id)).scalar()

    entity6 = Entity()
    instance10 = Instance(
        image=image5,
        entity=entity6,
        type_id=0,
        x=0,
        y=0,
        width=0,
        height=0,
        confidence=1,
    )

    test_session.add_all((entity6, instance10))
    test_session.commit()

    assert test_session.query(exists().where(Entity.id == entity6.id)).scalar()
    assert test_session.query(
        exists().where(
            and_(Instance.entity_id == entity6.id, Instance.image_id == image5.id)
        )
    ).scalar()


def test_transfer_entity(test_session: Session):
    image5 = Image(
        path="/img5.jpg", datetime=datetime.fromisoformat("2026-06-26T12:16:27Z")
    )
    entity6 = Entity()
    instance10 = Instance(
        image=image5,
        entity=entity6,
        type_id=0,
        x=0,
        y=0,
        width=0,
        height=0,
        confidence=1,
    )

    test_session.add_all((image5, entity6, instance10))
    test_session.commit()

    entity6_id = entity6.id
    instance10.entity = entity5

    test_session.add(instance10)
    test_session.commit()

    instances: List[Instance] = entity5.get_instances(test_session)
    assert len(instances) == 2
    assert instances[0].image_id == image4.id
    assert instances[1].image_id == image5.id

    assert not test_session.query(
        exists().where(Instance.entity_id == entity6_id)
    ).scalar()
    assert not test_session.query(exists().where(Entity.id == entity6_id)).scalar()
