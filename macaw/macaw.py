# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:22:31 2021

Contains the Macaw class, the Macaw_optimus function, and the smiles_clean 
function.

@author: Vincent
"""


import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMHFPFingerprint
from rdkit.Chem.rdmolops import LayeredFingerprint
from sklearn.manifold import Isomap
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_score
from functools import partial


class Macaw:
    __version__ = 'alpha_9tf'
    __author__ = 'Vincent Blay'
    def __init__(
        self,
        smiles,
        Nlndmk=50,
        fptype='Morgan2',
        metric='Tanimoto',
        lndmk_idx=[],
        Y=[],
        Yset=10,
        edim=15,
        method='MDS',
    ):
        """
        Class providing Macaw numeric embeddings of molecules.

        ...

        Attributes
        ----------

        _Nlndmk : int, optional
            Desired number of landmark molecules to use. Defaults to 50.

        _fptype : str, optional
            Desired type of fingerprint to use to characterize molecules.
            Options include 'RDK5', 'RDK7', 'Morgan2', 'Morgan3',
            'featMorgan2', 'featMorgan3,'Avalon','MACCS', 'atompairs',
            'torsion', 'pattern', 'secfp', and 'layered'. Combinations can
            also be specified with the '+' symbol, e.g. 'RDK5+MACCS'.
            Defaults to 'Morgan2'.

        _metric : str, optional
            Distrance metric used to measure similarity between molecular
            fingerprints. Options include 'Tanimoto', 'dice', 'cosine',
            'Sokal', 'Kulczynski', 'Mcconnaughey', 'Braun-Blanquet',
            'Rogot-Goldberg', 'asymmetric', and 'Manhattan'. Defaults to
            'Tanimoto'.

        _lndmk_idx : numpy.ndarray, optional
            Array indicating the `smiles` indices to be used as landmarks.

        _Y : numpy.ndarray, optional
            Array containing the property of interest for each molecule in the
            smiles input.

        _Yset : str or int, optional
            Specifies how to use the input in `Y`, if provided.
            Options include 'highest' and 'lowest'. If an integer is provided,
            it will use uniform sampling of landmarks after splitting the
            molecules in `Yset` bins. Defaults to 10.

        _edim : int, optional
            Number of dimensions for the embedding. Defaults to 15.

        _method : str, optional
            Method to use for the projection. Options available are 'MDS' and
            'isomap'. Defaults to 'MDS'.

        Methods
        -------

        transform(smiles : list, optional)
            Returns the Macaw embedding for the molecules provided in SMILES
            format. If `smiles`is not provided, it will embed the molecules
            that were supplied when the Macaw object was created.

        set_
            Setter methods are available for the different attributes. See
            Notes.


        Notes
        -------

        No attribute should be modified directly. Instead, setter methods are
        implemented for the different attributes, namely `set_Nlndmk()`,
        `set_lnmk_idx()`, `set_fptype()`, `set_metric()`, `set_Y()`,
        `set_Yset()`, `set_edim()`, and `set_method()`.

        If the `Y` attribute is set, then this information will be used in the
        choice of the landmarks. If `Yset` is an integer, then the dataset will
        be split in `Yset` bins according to `Y` and landmarks will be sampled
        from the bins with equal probability. If `Yset` is set to 'highest' or
        'lowest', then the landmarks will be the molecules with the highest or
        lowest `Y` values, respectively.


        """
        smiles = list(smiles)
        Y = np.array(Y)
        self.__mols, self.__bad_idx = self._smiles_to_mols(smiles, bad_idx=True)
        if Nlndmk > len(self.__mols):
            print(
                f"Not enough molecules provided. `Nlndmk` has been set to {len(self.__mols)}"
            )
            Nlndmk = len(self.__mols)
        self._Nlndmk = Nlndmk
        self._fptype = fptype.lower().replace(' ', '')
        self._metric = metric.lower().replace(' ', '')
        if len(Y) == len(smiles):
            mask = np.ones(len(Y), bool)
            mask[self.__bad_idx] = False
            self._Y = Y[mask]
        else:
            if len(Y) != 0:
                raise IOError(
                    "Error: inputs Y and smiles must have the same length."
                )
            self._Y = []
        self._Yset = Yset
        self._edim = edim
        if method.lower() not in ['mds', 'isomap']:
            raise IOError(
                f"Error: Invalid method {method}. Methods available are 'mds' and 'isomap'."
            )
            method = 'mds'
        self._method = method.lower()
        self._lndmk_idx = lndmk_idx

        # These attributes need not be visible or modified by the user,
        # we make them private
        self.__refD = []
        self.__LndS = []
        self.__refps = []

        if len(lndmk_idx) == 0:
            self.__lndmk_choice()
            self.__refps_update()
            self.__refD_update()
            self.__safe_lndmk_embed()
        else:
            self.set_lndmk_idx(lndmk_idx)
            # This includes refps_update, refD_update, and safe_lndmk_embed

    # Setter methods

    def set_Nlndmk(self, Nlndmk):
        if Nlndmk > len(self.__mols):
            print(
                f"Not enough molecules provided. `Nlndmk` has been set to {len(self.__mols)}."
            )
            Nlndmk = len(self.__mols)

        self._Nlndmk = Nlndmk
        self.__lndmk_choice()
        self.__refps_update()
        self.__refD_update()
        self.__safe_lndmk_embed()

    def set_lndmk_idx(self, lndmk_idx):
        # We take into account if there were any bad_indices in the original
        # dataset provided

        lndmk_idx = np.sort(lndmk_idx)
        bad_idx = self.__bad_idx

        max_idx = len(bad_idx) + len(self.__mols)
        for i in lndmk_idx:
            if i > max_idx - 1:
                raise IOError(f"Index {i} exceeds the number of molecules available.")
            if i < 0:
                raise IOError(f"Index {i} is not valid: indices must be positive integers.")
            if i in bad_idx:
                raise IOError(f"Invalid SMILES in position {i}.")

        # We are not storing bad smiles/molecules, so we shift the indices
        # accordingly to use the landmarks that the user means
        for i in bad_idx:
            lndmk_idx[lndmk_idx > i] -= 1

        self._lndmk_idx = lndmk_idx
        self._Nlndmk = len(lndmk_idx)
        self.__refps_update()
        self.__refD_update()
        self.__safe_lndmk_embed()

    def set_fptype(self, fptype):
        self._fptype = fptype.lower().replace(' ', '')
        self.__refps_update()
        self.__refD_update()
        self.__safe_lndmk_embed()

    def set_metric(self, metric):
        self._metric = metric.lower().replace(' ', '')
        # self.__refps_update()
        self.__refD_update()
        self.__safe_lndmk_embed()

    def set_Y(self, Y):
        Y = np.array(Y)
        long = len(self.__mols) + len(self.__bad_idx)
        if len(Y) == long:
            mask = np.ones(len(Y), bool)
            mask[self.__bad_idx] = False
            self._Y = Y[mask]
        else:
            if len(Y) != 0:
                raise IOError(f"Input Y has length {len(Y)} but it should be 0 or {long}.")
            self._Y = []
        self.__lndmk_choice()
        self.__refps_update()
        self.__refD_update()
        self.__safe_lndmk_embed()

    def set_Yset(self, Yset):
        self._Yset = Yset
        self.__lndmk_choice()
        self.__refps_update()
        self.__refD_update()
        self.__safe_lndmk_embed()

    def set_edim(self, edim):
        self._edim = edim
        self.__safe_lndmk_embed()

    def set_method(self, method):
        if method.lower() not in ['mds', 'isomap']:
            raise IOError(f"Unknown method {method}.")
        self._method = method.lower()
        self.__safe_lndmk_embed()

    # Main functions for the embedding

    def __lndmk_choice(self):

        mols = self.__mols
        Nlndmk = self._Nlndmk
        Y = self._Y
        Yset = self._Yset

        # Let us first extract the landmark fingerprints
        # If Y is not provided, pick the landmarks randomly
        lenY = len(Y)
        if lenY != len(mols):
            lndmk_idx = np.random.choice(range(len(mols)), Nlndmk, replace=False)
            self._Y = []

        else:
            if Yset == 'highest':  # gets the landmarks from the top
                lndmk_idx = np.argpartition(Y, -Nlndmk)[-Nlndmk:]
            elif Yset == 'lowest':  #get the landmarks from the bottom
                lndmk_idx = np.argpartition(Y, Nlndmk)[:Nlndmk]
            else:  # Yset is an integer
                try:
                    nbins = min(Yset, Nlndmk)
                except TypeError:
                    raise IOError("Invalid Yset argument.")
                Y_binned = np.floor(
                    (Y - min(Y)) / (1.00001 * (max(Y) - min(Y))) * nbins
                )

                proba = []
                for i in range(lenY):
                    proba.append(1.0 / sum(Y_binned == Y_binned[i]))
                proba = proba / sum(proba)
                lndmk_idx = np.random.choice(
                    range(len(mols)), Nlndmk, replace=False, p=proba
                )

        self._lndmk_idx = np.sort(lndmk_idx)

    def __refps_update(self):
        mols = self.__mols
        lndmk_idx = self._lndmk_idx
        remols = [mols[i] for i in lndmk_idx]
        # Equivalent to list(itemgetter(*lndmk_idx)(mols))

        refps = self.__fps_maker(remols)
        self.__refps = refps

    def __refD_update(self):
        refps = self.__refps
        D = self.__fps_distance(refps)
        self.__refD = D

    def __safe_lndmk_embed(self):
        # This function tries to call lndmk_embed and if it fails will reduce
        # edim to make it work.
        copy_edim = self._edim
        edim = self._edim

        if edim > self._Nlndmk:
            edim = self._Nlndmk

        for i in range(edim, 0, -1):  # will try down to edim = 1
            try:
                self._edim = i
                self.__lndmk_embed()
                break
            except ValueError:
                pass

        if self._edim < copy_edim:
            print(f"Not enough Nldnmk, edim has been set to {self._edim}")

    def __lndmk_embed(self):

        D = self.__refD
        method = self._method
        edim = self._edim

        n = len(D)

        if method == 'mds':
            D = D ** 2

            # Centering matrix
            H = -np.ones((n, n)) / n
            np.fill_diagonal(H, 1 - 1 / n)
            H = -H.dot(D ** 2).dot(H) / 2

            # Diagonalize
            evals, evecs = np.linalg.eigh(H)

            # Sort by eigenvalue in descending order
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]

            # Compute coordinates using positive-eigenvalued components only
            (w,) = np.where(evals > 0)
            if edim:
                arr = evals
                w = arr.argsort()[-edim:][::-1]
                if np.any(evals[w] < 0):
                    raise ValueError("Not enough positive eigenvalues for the selected edim.")

            if w.size == 0:
                raise ValueError("Matrix is negative definite.")

            V = evecs[:, w]
            Lh = V.dot(np.diag(1.0 / np.sqrt(evals[w]))).T
            means = np.mean(D, axis=1)
            LndS = (Lh, means)

        elif method == 'isomap':
            n_neighbors = int(np.ceil(1.4 * edim))
            LndS = Isomap(
                n_neighbors=n_neighbors, n_components=edim, metric='precomputed'
            )
            LndS.fit(D)

        self.__LndS = LndS

    def transform(self, qsmiles=[]):

        method = self._method
        LndS = self.__LndS

        qsmiles = list(qsmiles)

        # if qsmiles were not provided, then we will compute the distance
        # between self.__mols and the landmarks
        if len(qsmiles) == 0:
            mols = self.__mols
            bad_idx = self.__bad_idx
            lndmk_idx = self._lndmk_idx

            mols_no_lndmks = np.delete(mols, lndmk_idx)

            # In this case we do not need to recompute the distances between
            # landmarks because we already have them in self.__refD
            qfps = self.__fps_maker(mols_no_lndmks)
            D = self.__fps_distance(qfps)

            # Now we can insert the rows in D corresponding to the landmarks
            D_lndmks = self.__refD
            i = 0
            for row_idx in lndmk_idx:
                D = np.insert(D, row_idx, D_lndmks[i, :], axis=0)
                i += 1

        else:
            mols, bad_idx = self._smiles_to_mols(qsmiles, bad_idx=True)

            # We compute the fingerprints and distances for the query smiles
            qfps = self.__fps_maker(mols)
            D = self.__fps_distance(qfps)

        if method == 'mds':
            D = D ** 2

            Lh = LndS[0]
            means = LndS[1]
            if Lh.shape[1] != D.shape[1]:
                raise ValueError(
                    "Must provide distance of each point to every landmark"
                )
                return []
            D = D.T
            N = D.shape[1]
            Dm = D - np.tile(means, (N, 1)).T
            X = -Lh.dot(Dm) / 2.0
            X = X.T

        elif method == 'isomap':
            X = LndS.transform(D)

        #  We insert nan rows in the bad_idx positions if there were any
        for i in bad_idx:
            X = np.insert(X, i, np.nan, axis=0)

        return X

    # Helper functions
    def _smiles_to_mols(self, smiles, bad_idx=False):
        mols = []
        bad_ind = []
        for i in range(len(smiles)):
            m = MolFromSmiles(
                smiles[i], sanitize=True
            )  # let us implement a more robust strategy
            if m is not None:
                mols.append(m)
            else:
                print(f"Warning: Invalid SMILES in position {i}: {smiles[i]}")
                bad_ind.append(i)

        if bad_idx:
            return mols, bad_ind
        else:
            return mols

    def __fps_maker(self, mols):
        fptype = self._fptype

        switcher = {
            'morgan2': partial(
                AllChem.GetMorganFingerprintAsBitVect, radius=2, nBits=1024
            ),
            'morgan3': partial(
                AllChem.GetMorganFingerprintAsBitVect, radius=3, nBits=1024
            ),
            'rdk5': partial(Chem.RDKFingerprint, minPath=1, maxPath=5, fpSize=1024),
            'rdk7': partial(Chem.RDKFingerprint, minPath=1, maxPath=7, fpSize=2048),
            'featmorgan2': partial(
                AllChem.GetMorganFingerprintAsBitVect,
                radius=2,
                useFeatures=True,
                useChirality=True,
                nBits=2048,
            ),
            'featmorgan3': partial(
                AllChem.GetMorganFingerprintAsBitVect,
                radius=3,
                useFeatures=True,
                useChirality=True,
                nBits=2048,
            ),
            'maccs': rdMolDescriptors.GetMACCSKeysFingerprint,
            'avalon': pyAvalonTools.GetAvalonFP,
            'atompairs': rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect,
            'torsion': rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect,
            'pattern': partial(Chem.PatternFingerprint, fpSize=4096),
            'secfp': partial(rdMHFPFingerprint.MHFPEncoder(0, 0).EncodeSECFPMol),
            'layered': LayeredFingerprint,
        }

        fptypes = fptype.split('+')

        if len(fptypes) > 1:

            def F(mol, fptypes):
                n = 0
                keys = []
                for fptype_i in fptypes:
                    f = switcher[fptype_i]
                    fpi = f(mol)
                    for i in fpi.GetOnBits():
                        keys.append(i + n)

                    n += fpi.GetNumBits()

                fp = Chem.DataStructs.ExplicitBitVect(n)
                for i in keys:
                    fp[i] = 1

                return fp

            fps = [F(mol, fptypes) for mol in mols]

        else:  # len(fptypes)==1
            f = switcher[fptypes[0]]
            fps = list(map(f, mols))
            # Equivalent to fps = [f(mol) for mol in mols]

        return fps

    def __fps_distance(self, fps1=[]):
        metric = self.__metric_to_class()
        fps2 = self.__refps

        # Let us now compute the matrix of pairwise distances
        l1 = len(fps1)
        l2 = len(fps2)
        if l1 == 0:  # If fps1 is not provided, we return the symmetric matrix
            # of distances between refps
            S = np.zeros((l2, l2))
            np.fill_diagonal(S, 1)
            for i in range(l2 - 1):
                for j in range(i + 1, l2):
                    s = DataStructs.FingerprintSimilarity(
                        fps2[i], fps2[j], metric=metric
                    )
                    S[i, j] = s
                    S[j, i] = s
        else:
            S = np.zeros((l1, l2))
            for i in range(l1):
                for j in range(l2):
                    s = DataStructs.FingerprintSimilarity(
                        fps1[i], fps2[j], metric=metric
                    )
                    S[i, j] = s

        D = 1.0 - S
        return D

    def __metric_to_class(self):
        metric = self._metric

        if metric == 'tanimoto':
            r = DataStructs.TanimotoSimilarity
        elif metric == 'cosine':
            r = DataStructs.CosineSimilarity
        elif metric == 'dice':
            r = DataStructs.DiceSimilarity
        # elif metric =='russel': r = DataStructs.RusselSimilarity # non-commutative
        elif metric == 'sokal':
            r = DataStructs.SokalSimilarity
        elif metric == 'kulczynski':
            r = DataStructs.KulczynskiSimilarity
        elif metric == 'mcconnaughey':
            r = DataStructs.McConnaugheySimilarity
        # elif metric == 'tversky': r = lambda x, y: DataStructs.TverskySimilarity(x, y, 0.01, 0.99) # non-commutative
        elif metric == 'braun-blanquet':
            r = DataStructs.BraunBlanquetSimilarity
        elif metric == 'rogot-goldberg':
            r = DataStructs.RogotGoldbergSimilarity
        # elif metric == 'onbit': r = DataStructs.OnBitSimilarity # Same as Tanimoto
        elif metric == 'asymmetric':
            r = DataStructs.AsymmetricSimilarity
        elif metric == 'manhattan':
            r = DataStructs.AllBitSimilarity
        else:
            print(
                f"Warning: Invalid similarity metric {metric}. \
                    metric has been set to Tanimoto."
            )
            self._metric = "tanimoto"
            r = DataStructs.TanimotoSimilarity

        return r


# ----- End of Macaw class -----


def Macaw_optimus(
    smiles, y, fast=True, C=20.0, problem='auto', verbose=False, **kwargs
):
    """
    Function that identifies and recommends a Macaw embedding for a given
    problem. It does so by evaluating the performance of different embeddings
    as inputs to a support vector machine.

    ...

    Parameters
    ----------
    smiles : list
        List of molecules in SMILES format.

    y : numpy.ndarray
        Array containing the property of interest for each molecule in the
        smiles input.

    fast : bool, optional
        If set to True (default), evaluates only a subset of fingeprint types
        and distance metrics, works with a random sample of the dataset
        provided if it is large (>400 points), and uses 3 cv folds instead
        of 5.
        
    C : float , optional
        Regularization hyperparameter for the SVM. Defaults to 20.

    problem : str, optional
        Indicates whether it is a 'regression' or 'classification' problem. It
        determines if the model to use is a SVR or SVC. Defaults to 'auto',
        which will try to guess the problem type.

    verbose : bool, optional
        Prints intermediate scores for the different `fptype` and `metric`
        combinations.

    **kwargs : optional
        Allows to pass additional parameters to the Macaw class (other than
        `fptype`and `metric`).


    Returns
    -------
    Macaw
        Macaw object with the optimal settings identified.


    """
    smiles = list(smiles)
    y = np.array(y)

    leny = len(y)
    if leny != len(smiles):
        raise IOError(
            "`len(smiles)` = {len(smiles)} does not match `len(y)` = {leny}"
        )
    
    # If not specified, we will use the same Y argument for the individual
    # Macaw calls as the Macaw_optimus y argument.
    if 'Y' not in kwargs:
        kwargs['Y'] = y.copy()
    else:
        if len(kwargs['Y'] != leny):
            raise IOError(f"len(smiles) = {leny} does not match len(Y) = {len(kwargs['Y'])}")
    
    if problem == 'auto':
        tmp = set()
        tmp = tmp.union(y[0:100])
        if len(tmp) > 2:
            problem = 'regression'
        else:
            problem = 'classification'
        print(f"Problem type identified as {problem}")
        
    
    smiles_subset = smiles
    y_subset = y
    if fast:
        fptypes = [
            'morgan2',
            'morgan3',
            'featmorgan3',
            'maccs',
            'atompairs',
            'secfp',
            'torsion',
            'rdk7',
        ]
        metrics = ['tanimoto', 'cosine', 'dice', 'rogot-goldberg']
        cv = 3

        if leny > 400:
            np.random.seed(123)
            idx = np.random.choice(range(len(smiles)), 400, replace=False)
            
            smiles_subset = [smiles[i] for i in idx]
            # Equivalent to list(itemgetter(*idx)(smiles))
             
            y_subset = y[idx].copy()
            
            if len(kwargs['Y']) > 0:
                Y_copy = kwargs['Y'].copy()
                kwargs['Y'] = y_subset


    else:
        fptypes = [
            'morgan2',
            'morgan3',
            'rdk5',
            'rdk7',
            'featmorgan2',
            'featmorgan3',
            'maccs',
            'avalon',
            'atompairs',
            'torsion',
            'pattern',
            'secfp',
        ]
        metrics = [
            'tanimoto',
            'cosine',
            'dice',
            'sokal',
            'kulczynski',
            'mcconnaughey',
            'braun-blanquet',
            'rogot-goldberg',
            'asymmetric',
            'manhattan',
        ]

        cv = 5
        
    
    mcw = Macaw(smiles_subset, **kwargs)

    if problem == 'regression':
        epsilon = np.ptp(y) / 25.0
        f = SVR(kernel='rbf', C=C, epsilon=epsilon)
    else:
        f = SVC(kernel='rbf', C=C, gamma='scale')

    M = np.zeros((len(fptypes), len(metrics)))
    for i in range(len(fptypes)):
        mcw.set_fptype(fptypes[i])

        for j in range(len(metrics)):
            mcw.set_metric(metrics[j])

            x = mcw.transform()

            # splitters are instantiated with shuffle=False so the splits
            # will be the same across calls.
            M[i, j] = cross_val_score(f, x, y_subset, cv=cv).mean()

            if verbose:
                print(f"{fptypes[i]} + {metrics[j]}: {M[i,j]:0.3f}")

    # Torsion + sokal combination returns nan's. We set its score to 0.0
    np.nan_to_num(M, copy=False)
    max_i, max_j = np.unravel_index(M.argmax(), M.shape)

    print(
        f"Setting recommended combination: {fptypes[max_i]} + {metrics[max_j]}"
    )

    # Restore the backup if needed
    if fast & (leny > 400) & (len(kwargs['Y']) > 0):
        kwargs['Y'] = Y_copy
        
    # Now we embed all input using the optimal combination
    mcw = Macaw(smiles, fptype=fptypes[max_i], metric=metrics[max_j], **kwargs)

    return mcw


# ----- AUXILIARY FUNCTIONS -----


def smiles_clean(smiles, idx=False):
    smiles = list(smiles)
    ind = []
    for i in range(len(smiles)):
        m = MolFromSmiles(
            smiles[i], sanitize=True
        )  # let us implement a more robust strategy
        if m is not None:
            ind.append(i)
        else:
            print(
                f"Warning: Skipping invalid SMILES in position {i}: {smiles[i]}"
            )

    clean_smiles = [smiles[i] for i in ind]
    # Equivalent to clean_smiles = list(itemgetter(*ind)(smiles))

    if idx:
        return clean_smiles, ind
    else:
        return clean_smiles
