from dashboard.database.database import db
from dashboard.database.models import DetectionHistory


def save_detection(status, confidence, bottle_count):

    detection = DetectionHistory(
        status=status,
        confidence=confidence,
        bottle_count=bottle_count
    )

    db.session.add(detection)
    db.session.commit()


def get_detections():

    return DetectionHistory.query.order_by(
        DetectionHistory.id.desc()
    ).all()


def get_statistics():

    detections = DetectionHistory.query.all()

    total = len(detections)

    total_bottles = sum(d.bottle_count for d in detections)

    average_confidence = (
        sum(d.confidence for d in detections) / total
        if total > 0 else 0
    )

    latest = (
        detections[-1].detected_at
        if total > 0 else None
    )

    return (
        total,
        total_bottles,
        average_confidence,
        latest
    )


def get_chart_data():

    detections = DetectionHistory.query.order_by(
        DetectionHistory.id
    ).all()

    return [
        (
            d.detected_at,
            d.confidence,
            d.bottle_count
        )
        for d in detections
    ]