from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, RadioField, SelectField, BooleanField, PasswordField
from wtforms.validators import DataRequired, Length, InputRequired, Email, EqualTo, ValidationError
from .validators import IsValidSMILES

class SmilesPredictForm(FlaskForm):
    smiles = StringField('SMILES', validators=[DataRequired(), Length(min=1, max=500), IsValidSMILES()])
    endpoint = RadioField('Endpoint', validators=[InputRequired()])
    model = SelectField('Model', choices=['Choose model after selecting endpoint'], id='model')
    submit = SubmitField('Predict')


def create_SmilesPredictForm(endpoints):
    form = SmilesPredictForm()
    form.endpoint.choices = [(endpoint, endpoint) for endpoint in list(endpoints)]
    form.endpoint.validators = [InputRequired()]
    return form