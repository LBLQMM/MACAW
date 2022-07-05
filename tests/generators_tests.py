## Here are the unit tests

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from macaw import MACAW
from macaw import library_maker, library_evolver
import selfies as sf
from rdkit import Chem


@pytest.fixture
def RON_data():
    # Import RON data
    df = pd.read_csv("notebooks/data/data_RON.csv")  # Path from root directory as indicated by pytest.ini location
    smiles = df.Smiles
    Y = df.RON

    # Split smiles into training and test set
    smi_train, smi_test, y_train, y_test = train_test_split(smiles, Y, test_size=0.2, random_state=42)

    return [smi_train, smi_test, y_train, y_test]


def test_library_maker_len():
    n_gen=100
    smiles = library_maker(smiles=None, n_gen= n_gen, max_len=10, random_state=42)
    
    assert len(smiles)/n_gen > 0.5
    
    smiles2 = library_maker(smiles=None, n_gen= 2*n_gen, max_len=10, random_state=42)
    
    assert len(smiles2) > len(smiles)


def test_library_maker_selfies():
    
    smiles, selfies = library_maker(smiles=None, n_gen= 100, max_len=10, 
                                    return_selfies=True, random_state=42)

    assert len(smiles) == len(selfies)
    
    # Let us now test that the SELFIES and the SMILES match
    for smi, selfie in zip(smiles, selfies):
        a = Chem.CanonSmiles(smi)
        b = Chem.CanonSmiles(sf.decoder(selfie))
        assert a == b


def test_library_maker_maxlen():    
    max_lens=[5,10,20]
    for max_len in max_lens:
        smiles, selfies = library_maker(smiles=None, n_gen= 100, max_len=max_len, 
                                        return_selfies=True, random_state=42)
        
        max_len_generated = np.max([sf.len_selfies(s) for s in selfies])
        assert max_len_generated <= max_len
    

def test_library_maker_tokens(RON_data):    
    
    smi_train, smi_test, y_train, y_test = RON_data
    
    # get SELFIES tokens in input molecules
    selfies_train = [sf.encoder(smi) for smi in smi_train]
    tokens_train = sf.get_alphabet_from_selfies(selfies_train)
    
    # generate molecules
    _, selfies_gen = library_maker(smiles=smi_train, n_gen= 500, max_len=10, 
                                            return_selfies=True, full_alphabet=False,
                                            random_state=42)
    
    tokens_gen = sf.get_alphabet_from_selfies(selfies_gen) 
    
    # get robust SELFIES alphabet
    robust_alphabet = sf.get_semantic_robust_alphabet()
    
    # check that the tokens generated are in the robust alphabet
    assert tokens_gen.issubset(robust_alphabet)
    
    # check that the tokens generated are limited to those in smi_train
    assert tokens_gen.issubset(tokens_train)
   
    
def test_library_maker_full_alphabet(RON_data):    
    
    smi_train, smi_test, y_train, y_test = RON_data
    
    # generate molecules
    _, selfies_gen = library_maker(smiles=smi_train, n_gen= 500, max_len=10, 
                                            return_selfies=True, full_alphabet=False,
                                            random_state=42)
    
    tokens_gen = sf.get_alphabet_from_selfies(selfies_gen) 
    
    # generate molecules using entire robust alphabet
    _, selfies_gen = library_maker(smiles=None, n_gen= 500, max_len=10, 
                                            return_selfies=True, full_alphabet=True,
                                            random_state=42)
    
    tokens_gen_all = sf.get_alphabet_from_selfies(selfies_gen) 
    
    # check that more tokens are used with the full_alphabet option
    
    assert tokens_gen.issubset(tokens_gen_all)
    