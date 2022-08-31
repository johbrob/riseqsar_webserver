from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, RadioField, SelectField
from wtforms.validators import DataRequired, Length, InputRequired
from custom_wtform_validators import IsValidSMILES


class SmilesPredictForm(FlaskForm):
    smiles = StringField('SMILES', validators=[DataRequired(), Length(min=1, max=500), IsValidSMILES()])
    endpoint = RadioField('Endpoint')
    model = SelectField('Model', choices=['Choose model after selecting endpoint'], id='model')
    submit = SubmitField('Predict')


def create_SmilesPredictForm(endpoints):
    form = SmilesPredictForm()
    form.endpoint.choices = [(endpoint, endpoint) for endpoint in list(endpoints)]
    form.endpoint.validators = [InputRequired()]
    return form
