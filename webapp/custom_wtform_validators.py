import rdkit
import rdkit.Chem
from wtforms.validators import ValidationError


class IsValidSMILES(object):
    def __init__(self, message=None):
        if not message:
            message = 'Field is not recognized as a valid SMILES by RDKit'
        self.message = message

    def __call__(self, form, field):
        if rdkit.Chem.MolFromSmiles(field.data) is None:
            raise ValidationError(self.message)

