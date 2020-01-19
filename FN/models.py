from FN import db, login_manager
from flask_login import UserMixin
from datetime import datetime

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    email_id= db.Column(db.String(120))
    password= db.Column(db.String(150), nullable=False)
    email = db.relationship("Emails", back_populates ="user")
    received = db.Column(db.String(1500), nullable=True)
    spam = db.Column(db.String(1500), nullable=True)
    ham = db.Column(db.String(1500), nullable=True)
    def __repr__(self):
        return f"User('{self.email_id}','{self.received}','{self.spam}','{self.ham}')"
