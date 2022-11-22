from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_mail import Mail
from .config import Config
from dotenv import load_dotenv
load_dotenv()


TEMPLATE_FOLDER = '/templates'
STATIC_FOLDER = '/static'
UPLOAD_FOLDER = '/uploads'

#app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER, )

db = SQLAlchemy()
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = 'users.login'
login_manager.login_message_category = 'info'
mail = Mail()

def create_app(config_class=Config):
    app = Flask(__name__, static_url_path='/static')
    app.config.from_object(config_class)

    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)

    from .users.routes import users
    from .main.routes import main
    from .errors.handlers import errors
    app.register_blueprint(users)
    app.register_blueprint(main)
    app.register_blueprint(errors)

    return app