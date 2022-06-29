import pandas as pd
import pytest
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from macaw import MACAW

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
    mcw = MACAW(random_state=42)

    return mcw


def test_RON_transform(basic_macaw, RON_data):
    # Get model and data
    mcw = basic_macaw
    smi_train, smi_test, y_train, y_test = RON_data


    # Fits and transforms smiles into macaw embedding 
    mcw = MACAW(random_state=42)
    mcw.fit(smi_train)
    X_train = mcw.transform(smi_train)

    # Loads frozen demo X_train to compare with
    X_train_frozen = np.load("macaw/integration_tests/files/X_train.npy")  

    assert np.isclose(X_train,X_train_frozen,rtol=1e-03).all()
