## Here are the unit tests

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from macaw import MACAW, MACAW_optimus

@pytest.fixture
def RON_data():
    # Import RON data
    df = pd.read_csv("notebooks/data/data_RON.csv")  # Path from root directory as indicated by pytest.ini location
    smiles = df.Smiles
    Y = df.RON

    # Split smiles into training and test set
    smi_train, smi_test, y_train, y_test = train_test_split(smiles, Y, test_size=0.2, random_state=42)

    return [smi_train, smi_test, y_train, y_test]

@pytest.fixture
def basic_macaw():
    # Basic macaw model
    #mcw = MACAW(type_fp='atompairs', metric='rogot-goldberg', n_components=15, n_landmarks=100, random_state=57)
    mcw = MACAW(random_state=42)

    return mcw

def test_MACAW(basic_macaw):
    mcw = basic_macaw

    assert mcw._n_components  == 15
    assert mcw._type_fp       == 'morgan2'
    assert mcw._idx_landmarks == None
    assert mcw._resmiles      == []
    assert mcw._isfitted      == False

def test_fit_transform(basic_macaw, RON_data):
    # Get model and data
    mcw = basic_macaw
    smi_train, smi_test, y_train, y_test = RON_data

    # Do fit transform and obtain previously frozen data
    X_train = mcw.fit_transform(smi_train)
    X_train_frozen = np.load("macaw/integration_tests/files/X_train.npy")

    assert np.isclose(X_train,X_train_frozen,rtol=1e-03).all()


