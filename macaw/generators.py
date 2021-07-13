# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:27:53 2021

Contains the library_maker and library_evolver functions, the hit_finder 
and hit_finder2 functions, and some auxiliary functions.

@author: Vincent
"""

import numpy as np
import selfies as sf
from operator import itemgetter
from queue import PriorityQueue
from sklearn.neighbors import BallTree, DistanceMetric
from scipy.optimize import minimize
from rdkit import Chem


# ----- Molecular library generation functions -----


def library_maker(
    smiles,
    n_gen=20000,
    pad_to_len=0,
    noise_sd_factor=0.3,
    algorithm='position',
    efficient=True,
    return_selfies=False,
):
    """
    Generates molecules in a probabilistic manner from a list of input molecules.

    ...
    Parameters
    ----------
    smiles : list
        List of molecules in SMILES format.

    n_gen : int, optional
        Target number of molecules to be generated. The actual number of
        molecules returned can be lower. Defaults to 20000.

    pad_to_len : int, optional
        Maximum length of the molecules generated in SELFIES format.
        By default, the maximum length seen in the input molecules will be used.

    noise_sd_factor: float, optional
        Adjusts the level of noise being added to the SELFIES frequency counts.
        Defaults to 0.3.

    algorithm : str, optional
        Select to use 'position' (default) or 'transition' algorithm to compute
        the probability of sampling different SELFIES characters.

    efficient : bool, optional
        If true (default) the efficiency of molecule generation will be higher
        but molecules may tend to be longer than the input dataset.

    Returns
    -------
    List
        List containing the molecules generated in SMILES format.

    Notes
    -----
    Internally, molecules are generated as SELFIES. The molecules generated
    are filtered to remove SELFIES mapping to the same SMILES, as well as
    equivalent SMILES. The molecules returned are canonical SMILES.

    """
    # Let us convert the smiles to selfies onehot
    selfies = []
    maxlen = 0
    for smi in smiles:
        smi = smi.replace('.', '')
        smi = smi.replace('.', '')
        smi = smi.replace('[C@H]', 'C')
        smi = smi.replace('[C@@H]', 'C')
        smi = smi.replace('/C', 'C')
        smi = smi.replace('\\C', 'C')
        smi = smi.replace('/c', 'c')
        smi = smi.replace('\\c', 'c')

        try:
            s = sf.encoder(smi)
            if s is None:
                print(
                    f"Warning: SMILES {smi} is encoded as `None` and will be dropped."
                )
                # This is likely due to a space in the smiles
            else:
                selfies.append(s)
                maxlen = max(
                    maxlen, sf.len_selfies(s)
                )  # Stores the length of the longest SELFIES
        except MemoryError:
            print(f"Warning: SMILES {smi} could not be encoded as SELFIES.")
            # This may be due to the SELFIES encoding not finishing,
            # **MemoryError, which was happening for some symmetric molecules.

    if pad_to_len < 1:
        pad_to_len = maxlen

    alphabet = sf.get_alphabet_from_selfies(selfies)

    # Remove symbols from alphabet and columns of prob_matrix that
    # do not have a state-dependent derivation rule in the SELFIES package
    robust_symbols = sf.get_semantic_robust_alphabet()
    alphabet = alphabet.intersection(robust_symbols)

    alphabet = list(sorted(alphabet))
    alphabet.append('[nop]')

    len_alphabet = len(alphabet)

    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

    idx_list = []

    for selfi in selfies:
        try:
            idx = sf.selfies_to_encoding(
                selfi, vocab_stoi=symbol_to_idx, pad_to_len=pad_to_len, enc_type='label'
            )

            idx_list.append(idx[:pad_to_len])
        except KeyError:
            print(f"Warning: SELFIES {selfi} is not valid and will be dropped.")
            # This may be due to some character missing in the alphabet

    manysmiles = [None] * n_gen
    manyselfies = [None] * n_gen

    if algorithm.lower() == 'transition':

        trans_mat = np.zeros((len_alphabet, len_alphabet))
        start_mat = np.zeros(len_alphabet)

        for idx in idx_list:

            i_old = idx[0]
            start_mat[i_old] += 1

            for i in idx[1:]:
                trans_mat[i_old, i] += 1
                i_old = i

        if efficient:
            start_mat[-1] = 0
            trans_mat[
                :, -1
            ] = 0  # This will lower the chance of sampling a [nop] selfies character

        # Here we will add some noise to the prob matrix and normalize it
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_mat = trans_mat / row_sums

        noise_mean = 0  # noise_mean_factor * 1./len_alphabet
        noise_sd = noise_sd_factor * 1.0 / len_alphabet
        noise = np.random.normal(noise_mean, noise_sd, (len_alphabet, len_alphabet))
        trans_mat = abs(trans_mat + noise)

        row_sums = trans_mat.sum(axis=1, keepdims=True)
        trans_mat = trans_mat / row_sums

        start_mat = start_mat / start_mat.sum()
        noise = np.random.normal(noise_mean, noise_sd, len_alphabet)
        start_mat = abs(start_mat + noise)
        start_mat = start_mat / start_mat.sum()

        choices = [None] * pad_to_len
        range_alphabet = range(len_alphabet)
        range_1_pad_to_len = range(1, pad_to_len)
        for i in range(n_gen):
            if (i + 1) % 10000 == 0:  # progress indicator
                print(i + 1)

            choices[0] = np.random.choice(range_alphabet, size=1, p=start_mat)[0]
            for j in range_1_pad_to_len:
                idx = choices[j - 1]  # return the corresponding row of probabilities
                choices[j] = np.random.choice(
                    range_alphabet, size=1, p=trans_mat[idx, :]
                )[0]

            # Let us obtain the corresponding selfies and smiles
            selfies = ''.join(itemgetter(*choices)(alphabet))
            # Equivalent to selfies = ''.join([alphabet[i] for i in choices])

            smiles = sf.decoder(selfies)

            # And let us save the molecule
            manysmiles[i] = smiles
            manyselfies[i] = selfies

    elif algorithm.lower() == 'position':

        prob_matrix = np.zeros((pad_to_len, len_alphabet))

        for idx in idx_list:
            for i in range(pad_to_len):
                prob_matrix[i, idx[i]] += 1

        if efficient:
            prob_matrix[:, -1] = 0

        # Here we will add some noise to the prob matrix and normalize it
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        prob_matrix = prob_matrix / row_sums

        noise_mean = 0  # noise_mean_factor * 1./len_alphabet
        noise_sd = noise_sd_factor * 1.0 / len_alphabet
        noise = np.random.normal(noise_mean, noise_sd, (pad_to_len, len_alphabet))
        prob_matrix = abs(prob_matrix + noise)

        # let us normalize the prob_matrix row-wise again
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        prob_matrix = prob_matrix / row_sums

        c = prob_matrix.cumsum(axis=1)

        # End of computing the probability matrix c

        # Let us now start generating random molecules based on c

        for i in range(n_gen):
            if i % 10000 == 0:  # progress indicator
                print(i)

            u = np.random.rand(len(c), 1)
            choices = (u < c).argmax(axis=1)

            # Let us obtain the corresponding selfies and smiles
            selfies = ''.join(
                itemgetter(*choices)(alphabet)
            )  # Equivalent to selfies = ''.join([alphabet[i] for i in choices])
            smiles = sf.decoder(selfies)

            # And let us save the molecule
            manysmiles[i] = smiles
            manyselfies[i] = selfies

    else:
        raise IOError("Unknown algorithm: {algorithm}.")

    # Let us now clean the generated library

    # Multiple selfies can be synonyms of the same smiles
    manysmiles, idx = np.unique(manysmiles, return_index=True)
    manysmiles = list(manysmiles)
    manyselfies = [manyselfies[i] for i in idx]  # Access multiple elements of list

    for i in range(len(manysmiles)):
        # Vincent's patches to SELFIES
        # I do not want the triple bond S#C being available
        manysmiles[i] = manysmiles[i].replace('S#C', 'S=C')
        manysmiles[i] = manysmiles[i].replace('C#S', 'C=S')
        manysmiles[i] = manysmiles[i].replace('[SH]#C', 'S=C')
        manysmiles[i] = manysmiles[i].replace('C#[SH]', 'C=S')

        # Several smiles can correspond to the same canonical smiles
        manysmiles[i] = Chem.CanonSmiles(manysmiles[i])

    manysmiles, idx = np.unique(manysmiles, return_index=True)
    manysmiles = list(manysmiles)
    manyselfies = [manyselfies[i] for i in idx]  # Access multiple elments of a list

    if return_selfies:
        return manysmiles, manyselfies
    else:
        return manysmiles


def library_evolver(
    smiles,
    mcw,
    model,
    spec,
    k1=3000,
    k2=100,
    n_rounds=8,
    n_hits=10,
    algorithm='transition',
    efficient=False,
    **kwargs,
):
    """

    Recommends a list of molecules close to a desired specification by
    evolving increasingly focused libraries.

    ...
    Parameters
    ----------
    smiles : list
        List of molecules in SMILES format.

    mcw : Macaw
        Embedder to featurize the `smiles` input.

    model :
        Function that takes as input the features produced by the
        Macaw embedder `mcw` and returns a scalar (predicted property).

    spec : float
        Target specification that the recommended molecules should be close to.

    k1 : int, optional
        Target number of molecules to be generated in each intermediate
        library. Defaults to 3000.

    k2 : int, optional
        Numer of molecules that should be selected and carried over from an
        intermediate library to the next round. Defaults to 100.

    n_rounds : int, optional
        Number of iterations for the library generation and selection process.
        Defaults to 8.

    n_hits : int, optional
        Number of recommended molecules to return as output.

    Returns
    -------
    list
        List containing the molecules recommended in SMILES format.

    np. array
        Array containing the predicted property values for each output
        molecule according to the model provided.

    Notes
    -----
    This function makes extensive use of the `library_maker` function. See
    the `library_maker` function help for information on additional
    parameters.

    """

    smiles = list(smiles)
    
    if not callable(mcw):
        try:
            mcw = mcw.transform
        except AttributeError:
            raise IOError('mcw input is not callable.')
    
    if not callable(model):
        try:
            model = model.predict
        except AttributeError:
            raise IOError('model input is not callable.')
    
    
    X = mcw(smiles)
    Y_old = model(X)

    for i in range(n_rounds):
        print(f"\nRound {i+1}")
        smiles_lib = library_maker(
            smiles,
            n_gen=k1,
            algorithm=algorithm,
            efficient=efficient,
            return_selfies=False,
            **kwargs,
        )

        X = mcw(smiles_lib)
        Y_lib = model(X)

        # We want to carry over the best molecules from the previous round
        smiles_lib += smiles  # concatenates lists
        smiles_lib, idx = np.unique(smiles_lib, return_index=True)
        smiles_lib = list(smiles_lib)

        # Append Y_old
        Y = np.concatenate((Y_lib, Y_old))
        Y = Y[idx]

        idx = find_Knearest_idx(spec, Y, k=k2)
        smiles = [smiles_lib[i] for i in idx]  # Access multiple elements of a list
        Y_old = Y[idx]

    # Return best molecules
    idx = find_Knearest_idx(spec, Y_old, k=n_hits)
    smiles = [smiles[i] for i in idx]  # Access multiple elements of a list
    Y = Y_old[idx]

    return smiles, Y


def hit_finder(X_lib, model, spec, X=[], Y=[], n_hits=10, k1=5, k2=25, p=1, n_rounds=1):
    """
    Identifies promising hit molecules from a library according to a property
    specification.

    ...
    Parameters
    ----------
    X_lib : numpy.ndarray
        Array containing the Macaw embedding of a library of molecules.
        It can be generated with the `Macaw_proj` function.

    model : function
        Function that predicts property values given instances from `X_lib`.

    spec : float
        Desired property value specification.

    X : numpy.ndarray, optional
        Array containing the Macaw embedding of known molecules.
        It can be generated with the Macaw `transform` method.

    Y : numpy.ndarray, optional
        Array containing the property values for the known molecules.

    n_hits : int, optional
        Number of desired hit molecules to be output. Defaults to 10.

    k1 : int, optional
        Number of initial seed molecules to be carried in the search.
        Defaults to 5.

    k2 : int, optional
        Number of molecules per seed to be retrieved for evaluation.
        Defaults to 25.

    p : int or float, optional
        Minkowski norm to be used in the retrieval of molecules. If 0 < `p` < 1,
        then a V-distance is used. Defaults to 1 (Manhattan distance).

    n_rounds: int, optional
        Number of times the whole search will be iterated over. Defaults to 1.

    Returns
    -------
    List
        List of indices of the promising molecules found in `X_lib`.

    numpy.ndarray
        Array of property values predicted for the hit molecules using the
        model supplied.

    Notes
    -----
    The function uses an heuristic search to identify molecules close to the
    desired specification across the library.

    If `X`and `Y` are provided, it first takes the `k1` known
    molecules closest to the specification to guide the retrieval of the `k2`
    closest molecules in the Macaw embedding space (according to a `p`-norm).
    This process is done using a sklearn BallTree structure. The `k1`x`k2`
    molecules retrieved are then evaluated using the model provided (`model`).
    If `n_rounds` = 1 (default), the indices of the `n_hits` molecules
    closest to the specification are finally returned to the user.
    If `n_rounds` > 1, then the `k1` molecules closest to the specification
    are used to initiate another retrieval round.

    The actual number of molecules being evaluated can be smaller than
    `k1`x`k2` if there is overlap between the list of molecules returned from
    different seeds.

    If a `p` value is provided such that 0 < `p` < 1, then V-distance
    is used. This can be regarded as a weighted version of Manhattan distance,
    see publication for details.

    """
    
    if not callable(model):
        try:
            model = model.predict
        except AttributeError:
            raise IOError('model input is not callable.')
    
    
    if len(X) > k1:
        if len(Y) == len(X):
            idx_seed = find_Knearest_idx(spec, Y, k=k1)
        else:
            if len(Y) != 0:
                print(
                    "Warning: Y and X must have the same length. Input Y \
                        will be ignored."
                )
            idx_seed = np.random.choice(range(len(X)), k1, replace=False)
        Xseed = X[idx_seed]

    else:
        idx_seed = np.random.choice(range(len(X_lib)), k1, replace=False)
        Xseed = X_lib[idx_seed]

    if type(p) not in [int, float]:
        raise IOError("Invalid argument type, p must be an integer.")

    if p == 1:
        dt = DistanceMetric.get_metric('manhattan')
    elif p < 1:
        dt = DistanceMetric.get_metric('pyfunc', func=vdistance, p=p)
    else:
        dt = DistanceMetric.get_metric('minkowski', p=p)  # p > 1

    tree = BallTree(X_lib, metric=dt)

    for i in range(n_rounds):

        dist, idx = tree.query(Xseed, k=k2)

        idx = [item for sublist in idx for item in sublist]  # flatten list
        idx = np.unique(idx)

        X_hits = X_lib[idx]

        Y_hits_pred = model(X_hits)

        # This is only relevant if we are to iterate
        if i < (n_rounds - 1):
            newidx = find_Knearest_idx(spec, Y_hits_pred, k=k1)
            newidx = idx[newidx]
            Xseed = X_lib[newidx]

    ind = find_Knearest_idx(spec, Y_hits_pred, k=n_hits)  # ind in idx

    idx = idx[ind]  # idx in library
    idx = list(idx)
    Y_hits_pred = Y_hits_pred[ind]

    return idx, Y_hits_pred


def hit_finder2(X_lib, model, spec, X=[], Y=[], n_hits=10, k1=25, k2=5, p=2):
    """
    Identifies promising hit molecules from a library according to a property
    specification. Best suited for smooth embeddings like Macaw.

    ...
    Parameters
    ----------
    X_lib : numpy.ndarray
        Array containing the Macaw embedding of a library of molecules.
        It can be generated with the `Macaw_proj` function.

    model : function
        Function that predicts property values given instances from `X_lib`.

    spec : float
        Desired property value specification.

    mcw : Macaw object, optional
        Macaw object used to embed molecules in `X_lib`.

    n_hits : int, optional
        Number of desired hit molecules to be output. Defaults to 10.

    k1 : int, optional
        Number of initial seed molecules to be used in the search.
        Defaults to 25.

    k2 : int, optional
        Number of molecules per seed to be retrieved for evaluation.
        Defaults to 5.

    p : int or float, optional
        Minkowski norm to be used in the retrieval of molecules. If 0 < `p` < 1,
        then a V-distance is used. Defaults to 1 (Manhattan distance).


    Returns
    -------
    List
        List of indices of the promising molecules found in `X_lib`.

    numpy.ndarray
        Array of property values predicted for the hit molecules.

    Notes
    -----
    The function solves the model for the desired specification using
    Powell's algorithm implemented in scipy's fzero using `k1` starting
    seeds. Then, it retrieves `k2` molecules in `X_lib` close to each solution
    using sklearn's BallTree.

    If `mcw` is provided, it will use `k1` landmark molecules as seeds, which
    may offer better diversity of solutions.

    The actual number of molecules being evaluated can be smaller than
    `k1`x`k2` if there is overlap between the list of molecules returned from
    different seeds.

    If a `p` value is provided such that 0 < `p` < 1, then V-distance
    is used. This can be regarded as a weighted version of Manhattan distance,
    see publication for details.

    """
    
    if not callable(model):
        try:
            model = model.predict
        except AttributeError:
            raise IOError('model input is not callable.')
    
    
    if len(X) > k1:
        if len(Y) == len(X):
            idx_seed = find_Knearest_idx(spec, Y, k=k1)
        else:
            if len(Y) != 0:
                print(
                    "Warning: Y and X must have the same length. Input Y will be ignored."
                )
            idx_seed = np.random.choice(range(len(X)), k1, replace=False)
        Xseed = X[idx_seed]

    else:
        idx_seed = np.random.choice(range(len(X_lib)), k1, replace=False)
        Xseed = X_lib[idx_seed]

    obj_function = (
        lambda x: (model([x]) - spec) ** 2
    )  # for use with minimize: vector -> scalar

    z = []
    for i in range(k1):
        zi = minimize(
            obj_function,
            x0=Xseed[i],
            method='SLSQP',
            jac='2-point',
            options={'disp': False, 'ftol': 1e-4, 'maxiter': 20, 'eps': 1e-6},
        )
        z.append(zi.x)
    z = np.unique(np.array(z), axis=0)

    if type(p) not in [int, float]:
        raise IOError("Invalid argument type, p must be an integer.")

    if p == 1:
        dt = DistanceMetric.get_metric("manhattan")
    elif p < 1:
        dt = DistanceMetric.get_metric('pyfunc', func=vdistance, p=p)
    else:
        dt = DistanceMetric.get_metric('minkowski', p=p)  # p > 1

    tree = BallTree(X_lib, metric=dt)

    dist, idx = tree.query(z, k=k2)

    idx = [item for sublist in idx for item in sublist]  # flatten list
    idx = np.unique(idx)

    X_hits = X_lib[idx]

    Y_hits_pred = model(X_hits)

    ind = find_Knearest_idx(spec, Y_hits_pred, k=n_hits)  # ind in idx

    idx = idx[ind]  # idx in library
    idx = list(idx)
    Y_hits_pred = Y_hits_pred[ind]

    return idx, Y_hits_pred


# ----- AUXILIARY FUNCTIONS -----


def find_Knearest_idx(x, arr, k=1):
    """
    Finds the `k` nearest values to number `x` in unsorted array `arr` using a
    heap data structue.

    Adapted from https://www.geeksforgeeks.org/find-k-closest-numbers-in-an-unsorted-array/

    """

    n = len(arr)
    k = min(n, k)
    # Make a max heap of difference with
    # first k elements.
    pq = PriorityQueue()

    idx = []
    for i in range(k):
        pq.put((-abs(arr[i] - x), i))

    # Now process remaining elements
    for i in range(k, n):
        diff = abs(arr[i] - x)
        p, pi = pq.get()
        curr = -p

        # If difference with current
        # element is more than root,
        # then put it back.
        if diff > curr:
            pq.put((-curr, pi))
            continue
        else:

            # Else remove root and insert
            pq.put((-diff, i))

    # Print contents of heap.
    while not pq.empty():
        p, q = pq.get()
        idx.append(q)

    idx = np.array(idx)[np.argsort(arr[idx])]  # sort idx by arr value
    return idx


def vdistance(v1, v2, p=1):
    """
    Computes the V-distance between points v1 and v2.

    """
    v = np.sort(abs(v1 - v2))
    c = 1
    d = 0
    for el in v:
        d += c * el
        c *= p
    return d