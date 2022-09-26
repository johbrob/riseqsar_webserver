from flask import url_for, current_app
from flask_mail import Message
from .. import mail

def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message(subject='Password Reset Request',
                  sender=current_app.config['MAIL_USERNAME'],
                  recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_password', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made
'''
    mail.send(msg)