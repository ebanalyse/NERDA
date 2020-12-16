from NERDA.models import NERDA

# instantiate model.
model = NERDA()

def test_instantiate_NERDA():
    assert isinstance(model, NERDA)
