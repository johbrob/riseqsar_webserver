from wtforms.validators import ValidationError
from ..db_models import User

class NotTakenUsername(object):
    def __init__(self, message=None):
        if not message:
            message = 'That username is taken. Please choose a different one'
        self.message = message

    def __call__(self, form, username):
        if User.query.filter_by(username=username.data).first():
            raise ValidationError(self.message)


class NotTakenEmail(object):
    def __init__(self, message=None):
        if not message:
            message = 'That email is taken. Please choose a different one'
        self.message = message

    def __call__(self, form, email):
        if User.query.filter_by(email=email.data).first():
            raise ValidationError(self.message)


class EmailExists(object):
    def __init__(self, message=None):
        if not message:
            message = 'There is no account with that email. Check if email is correct. Otherwise consider register'
        self.message = message

    def __call__(self, form, email):
        if not User.query.filter_by(email=email.data).first():
            raise ValidationError(self.message)