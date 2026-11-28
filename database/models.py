from datetime import datetime

from database.database import db


class DetectionHistory(db.Model):

    __tablename__ = "detection_history"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    detected_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    status = db.Column(
        db.String(30),
        nullable=False
    )

    confidence = db.Column(
        db.Float,
        nullable=False
    )

    bottle_count = db.Column(
        db.Integer,
        nullable=False
    )