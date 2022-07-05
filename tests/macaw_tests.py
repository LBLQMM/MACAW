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
    X_train_frozen = np.load("tests/testdata/X_train.npy")
    
    assert X_train.dtype == float
    assert np.allclose(X_train,X_train_frozen,rtol=1e-03)


def test_shape_embedding(RON_data):
    N = [10, 15, 20]
    for n in N:
        mcw = MACAW(n_components=n)
        smi_train, smi_test, y_train, y_test = RON_data
    
        # Do fit transform and obtain previously frozen data
        X_train = mcw.fit_transform(smi_train)
    
        assert X_train.shape[1] == n
        assert X_train.shape[0] == len(smi_train)
        assert mcw._n_components == n


def test_shape_protection(RON_data):
    n_landmarks = 15
    N = [10, 15, 20]
    for n_components in N:
        mcw = MACAW(n_components=n_components, n_landmarks=n_landmarks)
        smi_train, smi_test, y_train, y_test = RON_data
    
        # Do fit transform and obtain previously frozen data
        X_train = mcw.fit_transform(smi_train)
    
        assert X_train.shape[1] <= n_landmarks
        assert mcw._n_landmarks == n_landmarks
        assert X_train.shape[1] == mcw._n_components
        

def test_invalid_smiles(basic_macaw, RON_data):
    test_smiles = ['CCCCCO', 'invalidtext', 'CC(CCC)CCOC']
    # Get model and data
    mcw = basic_macaw
    smi_train, smi_test, y_train, y_test = RON_data

    # Do fit transform and obtain previously frozen data
    mcw.fit(smi_train)
    X = mcw.transform(test_smiles)
    
    assert len(X) == len(test_smiles)
    assert all(np.isnan(X[1,:]))

    
def test_metric(RON_data):
    # Get model and data
    mcw = MACAW(metric="Manhattan", random_state=42)
    smi_train, smi_test, y_train, y_test = RON_data

    # Do fit transform and obtain previously frozen data
    X_train = mcw.fit_transform(smi_train)
    X_train_frozen = np.load("tests/testdata/X_train_manhattan.npy")
    
    assert X_train.dtype == float
    assert np.allclose(X_train,X_train_frozen,rtol=1e-03)

    
def test_set_metric(basic_macaw, RON_data):
    # Get model and data
    mcw = basic_macaw
    smi_train, smi_test, y_train, y_test = RON_data
    
    # Get default embedding
    X_train = mcw.fit_transform(smi_train)

    # Change to use cosine metric
    mcw.set_metric("cosine")
    X_train_cosine = mcw.transform(smi_train)
    
    # Change back to original Tanimoto metric
    mcw.set_metric("Tanimoto")
    X_train_tanimoto = mcw.transform(smi_train)
    
    assert not np.allclose(X_train,X_train_cosine,rtol=1e-03)
    assert np.allclose(X_train,X_train_tanimoto,rtol=1e-03)
    
    
def test_set_landmarks(basic_macaw, RON_data):
    # Get model and data
    mcw = basic_macaw
    smi_train, smi_test, y_train, y_test = RON_data
    
    # Get default embedding
    X_train = mcw.fit_transform(smi_train)
    
    # Get landmarks
    idx = mcw._idx_landmarks

    # Change to use selected landmarks
    mcw = MACAW(random_state=42, idx_landmarks = np.arange(50)) # force same seed
    X_train2 = mcw.fit_transform(smi_train)
    
    # Change back to original landmarks
    mcw = MACAW(random_state=79, idx_landmarks = idx) # force different seed
    X_train3 = mcw.fit_transform(smi_train)
    
    assert not np.allclose(X_train,X_train2,rtol=1e-03)
    assert np.allclose(X_train,X_train3,rtol=1e-03)


def test_use_landmarks(basic_macaw, RON_data):
    # Get model and data
    mcw = basic_macaw
    smi_train, smi_test, y_train, y_test = RON_data
    
    # Get default embedding
    X_train = mcw.fit_transform(smi_train)
    
    # Get landmarks
    idx = mcw._idx_landmarks
    smi_train = list(smi_train)
    smi_landmarks = [smi_train[i] for i in idx]

    # Train embedder passing only landmarks
    mcw = MACAW(n_landmarks=len(smi_landmarks)) # force same seed
    mcw.fit(smi_landmarks)
    X_train2 = mcw.transform(smi_train)
    
    assert np.allclose(X_train,X_train2,rtol=1e-03)

    
def test_set_algorithm(basic_macaw, RON_data):
    # Get model and data
    mcw = basic_macaw
    smi_train, smi_test, y_train, y_test = RON_data
    
    # Get default embedding
    X_train = mcw.fit_transform(smi_train)

    # Change to use selected algorithm
    mcw.set_algorithm("isomap")
    X_train2 = mcw.fit_transform(smi_train)
    
    # Change back to original algorithm
    mcw.set_algorithm("MDS")
    X_train3 = mcw.fit_transform(smi_train)
    
    assert not np.allclose(X_train,X_train2,rtol=1e-03)
    assert np.allclose(X_train,X_train3,rtol=1e-03)
 
    
def test_MACAW_optimus_type(RON_data):
    # Get model and data
    smi_train, smi_test, y_train, y_test = RON_data
    mcw = MACAW_optimus(smi_train, y=y_train, random_state=42)
    
    assert isinstance(mcw, MACAW)


def test_MACAW_optimus_shape(RON_data):
    
    n_components=17
    
    # Get model and data
    smi_train, smi_test, y_train, y_test = RON_data
    mcw = MACAW_optimus(smi_train, y=y_train, random_state=42, 
                        n_components=n_components)
    
    X_test = mcw.transform(smi_test)
    
    assert mcw._n_components  == n_components
    assert mcw._isfitted  == True
    assert X_test.shape[0] == len(X_test)
    assert X_test.shape[1] == n_components