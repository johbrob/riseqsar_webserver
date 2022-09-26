from flask import current_app
from flask_login import UserMixin
from datetime import datetime, timezone, timedelta
import jwt
from . import db, login_manager


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    def get_reset_token(self, expires_seconds=1800):
        encoding = jwt.encode(payload={'user_id': self.id,
                               'exp': datetime.now(tz=timezone.utc) + timedelta(seconds=expires_seconds)},
                               key=current_app.secret_key,
                              algorithm="HS256")
        return encoding

    @staticmethod
    def verify_reset_token(token):
        try:
            user_id = jwt.decode(jwt=token,
                                 key=current_app.secret_key,
                                 algorithms="HS256")['user_id']
        except:
            return None
        return User.query.get(user_id)


    def __repr__(self):
        return  f"User('{self.username}', '{self.email}')"
