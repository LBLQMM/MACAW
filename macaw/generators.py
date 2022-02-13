# -*- coding: utf-8 -*-
"""
Part of the MACAW project.
Contains the library_maker, the library_evolver functions, the hit_finder,
and the hit_finder2 functions.

@author: Vincent Blay, 2021
"""

import numpy as np
from operator import itemgetter
from queue import PriorityQueue
from rdkit import Chem
import re
from scipy.optimize import minimize
import selfies as sf
from sklearn.neighbors import BallTree, DistanceMetric


# ----- Molecular library generation functions -----

def library_maker(
    smiles=None,
    n_gen=20000,
    max_len=0,
    p="exp",
    noise_factor=0.1,
    algorithm="position",
    full_alphabet="False",
    return_selfies=False,
    random_state=None,
):
    """Generates molecules in a probabilistic manner. The molecules generated
    can be fully random or be biased around the distribution of input
    molecules.

    :param smiles: List of molecules in SMILES format. If not provided, it
        will generate random molecules using the alphabet of robust SELFIES
        tokens.
    :type smiles: list, optional
    :param n_gen: Target number of molecules to be generated. The actual
        number of molecules returned can be lower. Defaults to 20000.
    :type n_gen: int, optional
    :param max_len: Maximum length of the molecules generated in SELFIES
        format. If 0 (default), the maximum length seen in the input molecules
        will be used.
    :type max_len: int, optional
    :param p: Controls the SELFIES length distribution of molecules being generated.
        Options include 'exp' (exponential distribution, default), 'empirical'
        (observed input distribution), and 'cumsum' (cumulative observed
        input distribution). If `p` is numeric, then a potential distribution
        of degree `p` is used. If `p` is an array, then each element is
        considered to be a weight for sampling molecules with length
        given by the corresponding index (range(1,len(p+1))).
    :type p: str, float or numpy.ndarray, optional
    :param noise_factor: Controls the level of randomness added to the SELFIES
        frequency counts. Defaults to 0.1.
    :type noise_factor: float, optional
    :param algorithm: Select to use 'position', 'transition' or 'dual'
        algorithm to compute the probability of sampling different SELFIES
        tokens. Defaults to 'position'.
    :type algorithm: str, optional
    :param full_alphabet: Enables the use of all robust tokens in the SELFIES
        package. If False (default), only makes use of the tokens present
        in the input `smiles`.
    :type full_alphabet: bool, optional
    :param return_selfies: If True, the ouptut will include both SMILES and
        SELFIES.
    :type return_selfies: bool, optional
    :param random_state: Seed to have the same subsampling and choice of
        landmarks across runs.
    :type random_state: int, optional

    :return: List containing the molecules generated in SMILES format. If
        `return_selfies` is set to True, it will return a tuple with two 
        lists containing the SMILES and SELFIES, respectively.
    :rtype: list or tuple

    .. note:: Internally, molecules are generated as SELFIES. The molecules
        generated are filtered to remove synonyms. The molecules returned are
        canonical SMILES.
    """

    if random_state is not None:
        np.random.seed(random_state)

    if smiles is None:
        return _random_library_maker(
            n_gen=n_gen, max_len=max_len, return_selfies=return_selfies, p=p
        )

    # Let us convert the SMILES to SELFIES
    selfies = []
    lengths = []
    for s in smiles:
        smi = s.replace(" ", "")
        smi = smi.replace(".", "")
        # The following deals with SMILES atoms in brackets, like [C@H]
        # The only exceptions allowed are for tokens of the nitro group
        # which are now robust in SELFIES 2.0
        for m in re.findall("\[.*?\]", s):
            if m not in ['[N+]', '[O-]']:
                smi = smi.replace(m, m[1].upper())
        smi = smi.replace("/", "")
        smi = smi.replace("\\", "") # cis/trans isomery
        
        try:
            selfie = sf.encoder(smi)
            if selfie is None:
                print(
                    f"Warning: SMILES {s} is encoded as `None` and "
                    "will be dropped."
                )
            else:
                selfies.append(selfie)
                lengths.append(sf.len_selfies(selfie))
        except:
            print(f"Warning: SMILES {s} could not be encoded as SELFIES.")
            # This may be due to the SELFIES encoding not finishing,
            # **MemoryError, which was happening for some symmetric molecules.
	        # It may also be due to the input SMILES violating some semantic 
	        # constraint, e.g. a Cl with two bonds.

    lengths, max_len = __lengths_generator(max_len, n_gen, p, lengths)
    
    # Let us obtain the relevant alphabet of SELFIES tokens
    alphabet = sf.get_semantic_robust_alphabet()
    if not full_alphabet:
        custom_alphabet = sf.get_alphabet_from_selfies(selfies)
        # Remove symbols from alphabet and columns of prob_matrix that
        # do not have a state-dependent derivation rule in the SELFIES package
        alphabet = alphabet.intersection(custom_alphabet)
    alphabet = list(sorted(alphabet))
    len_alphabet = len(alphabet)
    
    
    # Let us onehot-encode the SELFIES
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

    idx_list = []
    for selfi in selfies:
        try:
            idx = sf.selfies_to_encoding(
                selfi, vocab_stoi=symbol_to_idx, pad_to_len=0, enc_type="label"
            )
            if len(idx) > 0:
                idx_list.append(idx)
        except KeyError:
            print(f"Warning: SELFIES {selfi} is not valid and will be dropped.")
            # This may be due to some symbol missing in the alphabet
    
    # Now we have the onehot SELFIES as a list of lists in idx_list
    
    # Let us now generate the prob_matrix
    
    algorithm = algorithm.lower()

    if algorithm == "transition":

        prob_matrix = np.zeros((len_alphabet+1, len_alphabet))
        # I add an extra row for the probabilities of the first token (last row)
        
        for idx in idx_list:
            k = len_alphabet
            for i in idx:
                prob_matrix[k, i] += 1
                k = i

    elif algorithm == "position":

        prob_matrix = np.zeros((max_len, len_alphabet))

        for idx in idx_list:
            for i in range(min(len(idx), max_len)):
                prob_matrix[i, idx[i]] += 1
        
    elif algorithm == "dual":
        
        # Here prob_matrix is a 3D matrix
        # The dimensions indicate the position in the SELFIES word,
        # the previous SELFIES token (0 at the beginning),
        # and the current SELFIES token
        prob_matrix = np.zeros((max_len, len_alphabet, len_alphabet))
        
        for idx in idx_list:
            k = 0
            for i in range(min(len(idx), max_len)):
                prob_matrix[i, k, idx[i]] += 1
                k = idx[i]           
            
    else:
        raise IOError(f'Invalid algorithm: {algorithm}.')
    
    # So far we have generated prob_matrix containing counts
    
    # Here we add some noise to prob matrix and normalize it
    prob_matrix = __noise_adder(prob_matrix, noise_factor=noise_factor)
    
    # Next we will sample SELFIES using prob_matrix
    manyselfies = [None] * n_gen
    
    if algorithm == 'transition':
        range_alphabet = range(len_alphabet)
        for i in range(n_gen):
            if (i + 1) % 10000 == 0:  # progress indicator
                print(f'{i+1} molecules generated.')
    
            len_i = lengths[i]
            choices = [None] * len_i
            
            idx = len_alphabet # initial token is sampled from last row
            for j in range(len_i):
                prob_vector = prob_matrix[idx, :]
                choices[j] = np.random.choice(range_alphabet, size=1, 
                                              p=prob_vector)[0]
                idx = choices[j]
    
            # Let us obtain the corresponding SELFIES
            selfies = "".join(itemgetter(*choices)(alphabet))
            # Equivalent to selfies = ''.join([alphabet[i] for i in choices])
    
            # And let us save it
            manyselfies[i] = selfies
        
    elif algorithm == 'position':

        c = prob_matrix.cumsum(axis=1)
        # Let us now start generating molecules based on c

        for i in range(n_gen):
            if (i + 1) % 10000 == 0:  # progress indicator
                print(f'{i+1} molecules generated.')

            len_i = lengths[i]
            u = np.random.rand(len_i, 1)
            choices = (u < c[:len_i, :]).argmax(axis=1)

            # Let us obtain the corresponding SELFIES
            selfies = "".join(itemgetter(*choices)(alphabet))  
            # Equivalent to selfies = ''.join([alphabet[i] for i in choices])

            # And let us save it
            manyselfies[i] = selfies
    
    elif algorithm == 'dual':
        
        range_alphabet = range(len_alphabet)
        for i in range(n_gen):
            if (i + 1) % 10000 == 0:  # progress indicator
                print(f'{i+1} molecules generated.')
                
            len_i = lengths[i]
            choices = [None] * len_i
            k = 0
            for j in range(len_i):
               prob_vector = prob_matrix[j,k,:]
               choices[j] = np.random.choice(range_alphabet, size=1, p=prob_vector)[0]
               k = choices[j]
               
            # Let us obtain the corresponding SELFIES
            selfies = "".join(itemgetter(*choices)(alphabet))  
            # Equivalent to selfies = ''.join([alphabet[i] for i in choices])
            
            # And let us save it
            manyselfies[i] = selfies
    
    
    # Let us now convert to SMILES and clean the generated library
    manysmiles, manyselfies = __selfies_to_smiles(manyselfies)

    print(f"{len(manysmiles)} unique molecules generated.")

    if return_selfies:
        return manysmiles, manyselfies
    else:
        return manysmiles


def _random_library_maker(n_gen=20000, max_len=15, return_selfies=False, p="exp"):
    """Generates random molecules using robust SELFIES tokens"""
    # In order to ensure that we get molecules of all lengths up to max_len
    # I first choose the length of the molecule, and then draw the number of
    # SELFIES tokens accordingly. We will not be padding the resulting SELFIES

    # In this case we will randomly sample and append SELFIES tokens
    alphabet = sf.get_semantic_robust_alphabet()
    alphabet = list(sorted(alphabet))

    # Choose the length of the molecules
    lengths, max_len = __lengths_generator(max_len, n_gen, p)

    manyselfies = [None] * n_gen
    for i in range(n_gen):
        if (i + 1) % 10000 == 0:  # progress indicator
            print(f'{i+1} molecules generated.')
        selfies = np.random.choice(alphabet, size=lengths[i], replace=True)
        selfies = "".join(selfies)
        manyselfies[i] = selfies

    manysmiles, manyselfies = __selfies_to_smiles(manyselfies)

    print(f"{len(manysmiles)} unique molecules generated.")

    if return_selfies:
        return manysmiles, manyselfies
    else:
        return manysmiles


def __noise_adder(matrix, noise_factor=0.1):
    """Normalizes the input matrix row-wise and mixes it linearly with a
    uniform matrix"""
    # Here we will add some noise to the prob matrix and normalize it
    k = matrix.ndim-1
    row_sums = matrix.sum(axis=k, keepdims=True)
    row_sums[row_sums == 0] = 1  # prevent division by zero
    prob_matrix = matrix / row_sums

    B = np.ones(prob_matrix.shape) / prob_matrix.shape[k]  # uniform matrix
    prob_matrix = (1 - noise_factor) * prob_matrix + noise_factor * B

    # normalize prob_matrix row-wise again, although should not be necessary
    row_sums = prob_matrix.sum(axis=k, keepdims=True)
    prob_matrix = prob_matrix / row_sums

    return prob_matrix


def __selfies_to_smiles(manyselfies):
    """Converts an input list of SELFIES into canonical SMILES
    removing duplicates and synonyms"""
    manysmiles = [None] * len(manyselfies)
    for i, selfies in enumerate(manyselfies):
        smiles = sf.decoder(selfies)
        # Vincent's patches to SELFIES
        # I do not want the triple bond S#C being available
        smiles = smiles.replace("S#C", "S=C")
        smiles = smiles.replace("C#S", "C=S")
        smiles = smiles.replace("[SH]#C", "S=C")
        smiles = smiles.replace("C#[SH]", "C=S")

        # Several smiles can correspond to the same canonical smiles
        manysmiles[i] = Chem.CanonSmiles(smiles)

    manysmiles, idx = np.unique(manysmiles, return_index=True)
    
    manyselfies = [manyselfies[i] for i in idx]
    
    # It is possible that the empty string '' makes it through
    # given that, e.g., sf.decoder('[Ring1]') returns ''. Let us remove it:
    idx = np.where(manysmiles=='')[0]
    manysmiles = list(manysmiles)
    if len(idx)>0:
        del manysmiles[idx[0]]
        del manyselfies[idx[0]]
    
    return manysmiles, manyselfies


def __lengths_generator(max_len, n_gen, p, lengths=None):
    """Returns a list of n_gen molecule lengths up to max_len.
    If max_len is 0, then max_len will be obtained from p or lengths.
    p describes the probability of generating different length values."""
    if max_len <= 0:
        if isinstance(p, np.ndarray):
            max_len = len(p)
        else:
            max_len = max(lengths)
        print(f"max_len set to {max_len}.")

    if isinstance(p, (int, float)):
        p = np.power(range(max_len), p)
    elif isinstance(p, str):
        p = p.lower()
        if p == "exp":
            p = np.exp(range(max_len))
        elif p == "empirical":
            p = np.zeros(max_len)
            for i in lengths:
                p[i - 1] += 1
        elif p == "cumsum":
            # This makes drawing longer molecules at least as likely as shorter ones
            p = np.zeros(max_len)
            for i in lengths:
                p[i - 1] += 1
            p = p.cumsum()

        else:
            raise IOError(f"Invalid p input value: {p}.")

    p = p / sum(p)

    lengths = np.random.choice(range(1, max_len + 1), size=n_gen, replace=True, p=p)
    return lengths, max_len


def library_evolver(
    smiles,
    model,
    mcw=None,
    spec=0.0,
    k1=2000,
    k2=100,
    n_rounds=8,
    n_hits=10,
    max_len=0,
    max_len_inc=2,
    force_new=False,
    random_state=None,
    **kwargs
):
    """Recommends a list of molecules close to a desired specification by
    evolving increasingly focused libraries.

    :param smiles: List of molecules in SMILES format.
    :type smiles: list
    :param model: Function that takes as input the features produced by the
        MACAW embedder `mcw` and returns a scalar (predicted property).
        The model may also directly take SMILES as its input, in which case
        no embedder needs to be provided. The model must be able to take a 
        list of multiple inputs and produce the corresponding list of 
        predictions.
    :type model: function
    :param mcw: Embedder to featurize the `smiles` input into a representation
        compatible with `model`. If not provided, it will be
        assigned the unity function, and the model will have to take SMILES
        directly as its input.
    :type mcw: MACAW or function, optional
    :param spec: Target specification that the recommended molecules should
        match.
    :type spec: float
    :param k1: Target number of molecules to be generated in each intermediate
        library. Defaults to 3000.
    :type k1: int, optional
    :param k2: Numer of molecules that should be selected and carried over
        from an intermediate library to the next round. Defaults to 100.
    :type k2: int, optional
    :param n_rounds: Number of iterations for the library generation and
        selection process. Defaults to 8.
    :type n_rounds: int, optional
    :param n_hits: Number of recommended molecules to return.
    :type n_hits: int, optional
    :param max_len: Maximum length of the molecules generated in SELFIES
        format. If 0 (default), the maximum length seen in the input molecules
        will be used.
    :type max_len: int, optional
    :param max_len_inc: Maximum increment in SELFIES length from one round
        to the next. Defaults to 2.
    :type max_len_inc: int, optional
    :param force_new: Forces to return only SMILES not present in the `smiles` 
        input. Defaults to False.
    :type force_new: book, optional
    :param random_state: Seed to have the same subsampling and choice of
        landmarks across runs.
    :type random_state: int, optional

    :return: A tuple `(list, numpy.ndarray)`. The first element is the list of molecules
        recommended in SMILES format. The second element is an array with the predicted property
        values for each recommended molecule according to the `model` provided.
    :rtype: tuple

    .. seealso:: This function makes extensive use of the `library_maker`
        function. See the `library_maker` documentation for
        information on additional parameters.
    """
    if random_state is not None:
        np.random.seed(random_state)

    if smiles is None:
        if max_len == 0:
            max_len = 15
        smiles = _random_library_maker(
            n_gen=20000, max_len=max_len, return_selfies=False, **kwargs
        )

    smiles = list(smiles)

    if mcw is None:
        mcw = lambda x: x

    if not callable(mcw):
        try:
            mcw = mcw.transform
        except AttributeError:
            raise IOError("mcw input is not callable.")

    if not callable(model):
        try:
            model = model.predict
        except AttributeError:
            raise IOError("model input is not callable.")

    X = mcw(smiles)
    Y_old = model(X)
    smiles_lib_old = smiles

    for i in range(n_rounds):
        print(f"\nRound {i+1}")
        smiles_lib = library_maker(
            smiles_lib_old,
            n_gen=k1,
            max_len=max_len,
            return_selfies=False,
            **kwargs
        )

        X = mcw(smiles_lib)
        Y_lib = model(X)

        # We want to carry over the best molecules from the previous round

        # Append the old molecules and remove duplicates
        smiles_lib += smiles_lib_old  # concatenates lists
        smiles_lib, idx = np.unique(smiles_lib, return_index=True)
        smiles_lib = list(smiles_lib)

        Y = np.concatenate((Y_lib, Y_old))
        Y = Y[idx]

        # Select k2 best molecules
        idx = __find_Knearest_idx(spec, Y, k=k2)
        smiles_lib_old = [smiles_lib[i] for i in idx]
        Y_old = Y[idx]
        
        if i < n_rounds-1:

            # Compute max_len to use in next round
            # For this I take the longest 10 SMILES amongst the k2
            # compute their SELFIES length and add +1 to the longest
            lengths = [len(smi) for smi in smiles_lib_old]  # lengths of smiles
            idx = np.argpartition(lengths, -10)[-10:]  # indices of 10 longest smiles
            lengths = [
                sf.len_selfies(sf.encoder(smiles_lib_old[i])) for i in idx
            ]  # length of selfies
            max_len = max(lengths) + max_len_inc
            print(f"max_len set to {max_len}.")
        
            
    # Remove molecules already in the input, if requested
    if force_new:
        idx = [i for i, smi in enumerate(smiles_lib) if smi not in smiles]
        smiles_lib = [smiles_lib[i] for i in idx]
        Y = Y[idx]
    
    # Return best molecules
    idx = __find_Knearest_idx(spec, Y, k=n_hits)
    smiles = [smiles_lib[i] for i in idx]  # Access multiple elements of a list
    Y = Y[idx]

    return smiles, Y


def hit_finder(X_lib, model, spec, X=[], Y=[], n_hits=10, k1=5, k2=25, p=1, n_rounds=1):
    """
    Identifies promising hit molecules from a library according to a property
    specification.

    :param X_lib: Array containing the MACAW embeddings of a library of
        molecules. It can be generated with the MACAW `transform` method.
    :type X_lib: numpy.ndarray
    :param model: Function that predicts property values given instances from `X_lib`.
    :type model: function
    :param spec: Desired property value specification.
    :type spec: float
    :param X: Array containing the MACAW embedding of known molecules.
        It can be generated with the MACAW `transform` method.
    :type X: numpy.ndarray, optional
    :param Y: Array containing the property values for the known molecules.
    :type Y: list or numpy.ndarray, optional
    :param n_hits: Desired number of hit molecules to be returned.
        Defaults to 10.
    :type n_hits: int, optional
    :param k1: Number of initial seed molecules to be carried in the search.
        Defaults to 5.
    :type k1: int, optional
    :param k2: Number of molecules per seed to be retrieved for evaluation.
        Defaults to 25.
    :type k2: int, optional
    :param p: Minkowski norm to be used in the retrieval of molecules.
        If 0 < `p` < 1, then a V-distance is used. Defaults to 1 (Manhattan distance).
    :type p: int or float, optional
    :param n_rounds: Number of times the whole search will be iterated over.
        Defaults to 1.
    :type n_rounds: int, optional

    :return: A tuple `(list,numpy.ndarray)`. The first element is
        the list of indices of the hit molecules found in `X_lib`. The
        second element is an array of property values predicted for the
        hit molecules using the model supplied.
    :rtype: tuple

    .. note::
        The function uses an heuristic search to identify molecules close to the
        desired specification across the library.

        If `X`and `Y` are provided, it first takes the `k1` known
        molecules closest to the specification to guide the retrieval of the `k2`
        closest molecules in the MACAW embedding space (according to a `p`-norm).
        This process is done using a sklearn BallTree structure. The `k1` x `k2`
        molecules retrieved are then evaluated using the model provided (`model`).
        If `n_rounds` = 1 (default), the indices of the `n_hits` molecules
        closest to the specification are finally returned to the user.
        If `n_rounds` > 1, then the `k1` molecules closest to the specification
        are used to initiate another retrieval round.

        The actual number of molecules being evaluated can be smaller than
        `k1` x `k2` if there is overlap between the list of molecules returned from
        different seeds.

    .. seealso:: If a `p` value is provided such that 0 < `p` < 1, then V-distance
        is used. This can be regarded as a weighted version of Manhattan distance,
        see publication for details.
    """

    if not callable(model):
        try:
            model = model.predict
        except AttributeError:
            raise IOError("model input is not callable.")

    if len(X) > k1:
        if len(Y) == len(X):
            idx_seed = __find_Knearest_idx(spec, Y, k=k1)
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
        dt = DistanceMetric.get_metric("manhattan")
    elif p < 1:
        dt = DistanceMetric.get_metric("pyfunc", func=__vdistance, p=p)
    else:
        dt = DistanceMetric.get_metric("minkowski", p=p)  # p > 1

    tree = BallTree(X_lib, metric=dt)

    for i in range(n_rounds):

        dist, idx = tree.query(Xseed, k=k2)

        idx = [item for sublist in idx for item in sublist]  # flatten list
        idx = np.unique(idx)

        X_hits = X_lib[idx]

        Y_hits_pred = model(X_hits)

        # This is only relevant if we are to iterate
        if i < (n_rounds - 1):
            newidx = __find_Knearest_idx(spec, Y_hits_pred, k=k1)
            newidx = idx[newidx]
            Xseed = X_lib[newidx]

    ind = __find_Knearest_idx(spec, Y_hits_pred, k=n_hits)  # ind in idx

    idx = idx[ind]  # idx in library
    idx = list(idx)
    Y_hits_pred = Y_hits_pred[ind]

    return idx, Y_hits_pred


def hit_finder2(X_lib, model, spec, X=[], Y=[], n_hits=10, k1=25, k2=5, p=2):
    """
    Identifies promising hit molecules from a library according to a property
    specification. Best suited for smooth embeddings like MACAW.

    :param X_lib: Array containing the MACAW embedding of a library of molecules.
        It can be generated with the MACAW `transform` method.
    :type X_lib: numpy.ndarray
    :param model: Function that predicts property values given instances from
        `X_lib`.
    :type model: function
    :param spec: Desired property value specification.
    :type spec: float
    :param X: Array containing the MACAW embedding of known molecules.
        It can be generated with the MACAW `transform` method.
    :type X: numpy.ndarray, optional
    :param Y: Array containing the property values for the known molecules.
    :type Y: list or numpy.ndarray, optional
    :param n_hits: Number of desired hit molecules to be output. Defaults to 10.
    :type n_hits: int, optional
    :param k1: Number of initial seed molecules to be used in the search.
        Defaults to 25.
    :type k1: int, optional
    :param k2: Number of molecules per seed to be retrieved for evaluation.
        Defaults to 5.
    :type k2: int, optional
    :param p: Minkowski norm to be used in the retrieval of molecules.
        If 0 < `p` < 1, then a V-distance is used. Defaults to 1 (Manhattan distance).
    :type p: int or float, optional

    :return: A tuple `(list,numpy.ndarray)`. The first element is
        the list of indices of the hit molecules found in `X_lib`. The
        second element is an array of property values predicted for the
        hit molecules using the model supplied.
    :rtype: tuple

    .. note::
        The function solves the model for the desired specification using
        Powell's algorithm implemented in scipy's fzero using `k1` starting
        seeds. Then, it retrieves `k2` molecules in `X_lib` close to each solution
        using sklearn's BallTree.

        If `mcw` is provided, it will use `k1` landmark molecules as seeds, which
        may offer better diversity of solutions.

        The actual number of molecules being evaluated can be smaller than
        `k1` x `k2` if there is overlap between the list of molecules returned from
        different seeds.

    .. seealso:: If a `p` value is provided such that 0 < `p` < 1, then V-distance
        is used. This can be regarded as a weighted version of Manhattan distance,
        see publication for details.
    """

    if not callable(model):
        try:
            model = model.predict
        except AttributeError:
            raise IOError("model input is not callable.")

    if len(X) > k1:
        if len(Y) == len(X):
            idx_seed = __find_Knearest_idx(spec, Y, k=k1)
        else:
            if len(Y) != 0:
                print(
                    "Warning: Y and X must have the same length. Input Y "
                    "will be ignored."
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
            method="SLSQP",
            jac="2-point",
            options={"disp": False, "ftol": 1e-4, "maxiter": 20, "eps": 1e-6},
        )
        z.append(zi.x)
    z = np.unique(np.array(z), axis=0)

    if type(p) not in [int, float]:
        raise IOError("Invalid argument type, p must be an integer.")

    if p == 1:
        dt = DistanceMetric.get_metric("manhattan")
    elif p < 1:
        dt = DistanceMetric.get_metric("pyfunc", func=__vdistance, p=p)
    else:
        dt = DistanceMetric.get_metric("minkowski", p=p)  # p > 1

    tree = BallTree(X_lib, metric=dt)

    dist, idx = tree.query(z, k=k2)

    idx = [item for sublist in idx for item in sublist]  # flatten list
    idx = np.unique(idx)

    X_hits = X_lib[idx]

    Y_hits_pred = model(X_hits)

    ind = __find_Knearest_idx(spec, Y_hits_pred, k=n_hits)  # ind in idx

    idx = idx[ind]  # idx in library
    idx = list(idx)
    Y_hits_pred = Y_hits_pred[ind]

    return idx, Y_hits_pred


# ----- AUXILIARY FUNCTIONS -----

def __find_Knearest_idx(x, arr, k=1):
    """
    Finds the `k` nearest values to number `x` in unsorted array `arr` using a
    heap data structue.
    
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

    # Print contents of heap
    while not pq.empty():
        p, q = pq.get()
        idx.append(q)

    idx = np.array(idx)[np.argsort(arr[idx])]  # sort idx by arr value
    return idx


def __vdistance(v1, v2, p=1):
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
