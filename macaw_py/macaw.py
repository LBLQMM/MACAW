# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:22:31 2021

Contains the Macaw class and the Macaw_optimus function.

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
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_score


class Macaw:
    """
    Class providing Macaw numeric embeddings of molecules.

    ...

    Attributes
    ----------

    _n_landmarks : int, optional
        Desired number of landmark molecules to use. Defaults to 50.

    _type_fp : str, optional
        Desired type of fingerprint to use to characterize molecules.
        Options include 'RDK5', 'RDK7', 'Morgan2', 'Morgan3',
        'featMorgan2', 'featMorgan3,'Avalon','MACCS', 'atompairs',
        'torsion', 'pattern', 'secfp6', and 'layered'. Combinations can
        also be specified with the '+' symbol, e.g. 'RDK5+MACCS'.
        Defaults to 'Morgan2'.

    _metric : str, optional
        Distrance metric used to measure similarity between molecular
        fingerprints. Options include 'Tanimoto', 'dice', 'cosine',
        'Sokal', 'Kulczynski', 'Mcconnaughey', 'Braun-Blanquet',
        'Rogot-Goldberg', 'asymmetric', and 'Manhattan'. Defaults to
        'Tanimoto'.

    _idx_landmarks : numpy.ndarray, optional
        Array indicating the `smiles` indices to be used as landmarks.

    _Y : numpy.ndarray, optional
        Array containing the property of interest for each molecule in the
        smiles input.

    _Yset : str or int, optional
        Specifies how to use the input in `Y`, if provided.
        Options include 'highest' and 'lowest'. If an integer is provided,
        it will use uniform sampling of landmarks after splitting the
        molecules in `Yset` bins. Defaults to 10.

    _n_components : int, optional
        Number of dimensions for the embedding. Defaults to 15.

    _algorithm : str, optional
        Algorithm to use for the projection. Options available are 'MDS',
        'isomap', 'PCA', ICA', and 'FA'. Defaults to 'MDS'.

    Methods
    -------

    fit(smiles: list, Y: list, optional, n_landmarks: int, optional,
        idx_landmarks: list, optional, Yset: int or str, optional)

    transform(smiles : list)
        Returns the Macaw embedding for the molecules provided in SMILES
        format.

    fit_transform(smiles: list, Y: list, optional, n_landmarks: int,
        optional, idx_landmarks: list, optional, Yset: int or str,
        optional)


    set_type_fp(str)

    set_metric(str)

    set_algorithm(str)

    set_n_components(int)


    Notes
    -------

    No attribute should be modified directly. Instead, setter methods are
    available for the modifiable attributes: `set_type_fp()`,
    `set_metric()`, `set_Y()`, `set_n_components()`, and `set_algorithm()`.

    If the `Y` argument is provided during fitting, it will be used in the
    choice of the landmarks. If `Yset` is an integer, then the dataset will
    be split in `Yset` bins according to `Y` and landmarks will be sampled
    from the bins with equal probability. If `Yset` is set to 'highest' or
    'lowest', then the landmarks will be the `n_landmarks` molecules with
    the highest or lowest `Y` values, respectively.


    """
    __version__ = "alpha_11"
    __author__ = "Vincent Blay"

    def __init__(
        self, type_fp="Morgan2", metric="Tanimoto", n_components=15, algorithm="MDS"
    ):


        self._n_components = n_components
        self._type_fp = type_fp.lower().replace(" ", "")
        self._metric = metric.lower().replace(" ", "")

        algorithm = algorithm.lower().replace(" ", "")
        if algorithm not in ["mds", "isomap", "pca", "ica", "fa"]:
            raise IOError(
                (
                    f"Error: Invalid algorithm {algorithm}. See the "
                    "documentation for available algorithms."
                )
            )
        self._algorithm = algorithm

        self._n_landmarks = None
        self._idx_landmarks = []
        self._resmiles = []

        # These attributes need not be visible or modified by the user,
        # we make them private
        self.__remols = []
        self.__refps = []
        self.__refD = []
        self.__LndS = []

    def __repr__(self):
        return (
            f"Macaw_embedder(n_landmarks={self._n_landmarks}, "
            f"n_components={self._n_components}, type_fp={self._type_fp},"
            f" metric={self._metric}, algorithm={self._algorithm})"
        )

    def __len__(self):
        return self._n_components

    # Setter algorithms

    def set_type_fp(self, type_fp):
        self._type_fp = type_fp.lower().replace(" ", "")
        if self._n_landmarks is not None:
            self.__refps_update()
            self.__refD_update()
            self.__safe_lndmk_embed()

    def set_metric(self, metric):
        self._metric = metric.lower().replace(" ", "")
        # self.__refps_update()
        if self._n_landmarks is not None:
            self.__refD_update()
            self.__safe_lndmk_embed()

    def set_n_components(self, n_components):
        self._n_components = n_components
        if self._n_landmarks is not None:
            self.__safe_lndmk_embed()

    def set_algorithm(self, algorithm):
        algorithm = algorithm.lower().replace(" ", "")
        if algorithm not in ["mds", "isomap", "pca", "ica", "fa"]:
            raise IOError(f"Unknown algorithm {algorithm}.")
        self._algorithm = algorithm

        if self._n_landmarks is not None:
            self.__safe_lndmk_embed()
    
    # Main functions for the embedding
    
    def fit(self, smiles, n_landmarks=50, Y=[], Yset=10, idx_landmarks=[],
            random_state=None):

        smiles = list(smiles)

        if n_landmarks > len(smiles):
            print(
                f"Warning: requested n_landmarks={n_landmarks} but only "
                f"{len(smiles)} smiles provided. n_landmarks will be lower."
            )
        
        if len(idx_landmarks) == 0:
            idx_landmarks = self.__lndmk_choice(smiles, n_landmarks, Y, Yset,
                                                random_state)
        else:
            idx_landmarks = np.sort(idx_landmarks)
        
        resmiles = [smiles[i] for i in idx_landmarks]
        remols, bad_idx = self._smiles_to_mols(resmiles, bad_idx=True)
        
        if len(bad_idx) > 0:
            idx_landmarks = np.setdiff1d(idx_landmarks, bad_idx)
            print(f"n_landmarks has been set to {len(idx_landmarks)}")
        
        resmiles = [smiles[i] for i in idx_landmarks]
        
        self._idx_landmarks = idx_landmarks
        self._n_landmarks = len(idx_landmarks)
        self._resmiles = resmiles
        self.__remols = remols
        self.__refps_update()
        self.__refD_update()
        self.__safe_lndmk_embed()

    def transform(self, qsmiles):

        if self._n_landmarks is None:
            raise RuntimeError(
                "Macaw instance not fitted. Call "
                "'fit' with appropriate arguments before using"
                " this embedder."
            )

        qsmiles = list(qsmiles)

        mols, bad_idx = self._smiles_to_mols(qsmiles, bad_idx=True)

        # We compute the fingerprints and distances for the query smiles
        qfps = self.__fps_maker(mols)
        D = self.__fps_distance_to_refps(qfps)

        return self.__project(D, bad_idx)

    def fit_transform(self, qsmiles, n_landmarks=50, Y=[], Yset=10, 
                      idx_landmarks=[], random_state=None):

        qsmiles = list(qsmiles)
        self.fit(
            smiles=qsmiles,
            n_landmarks=n_landmarks,
            Y=Y,
            Yset=Yset,
            idx_landmarks=idx_landmarks,
            random_state=random_state
        )

        idx_landmarks = self._idx_landmarks

        smi_no_lndmks = np.delete(qsmiles, idx_landmarks)
        mols_no_lndmks, bad_idx = self._smiles_to_mols(smi_no_lndmks, bad_idx=True)

        # In this case we do not need to recompute the distances between
        # landmarks: we already have them in self.__refD
        qfps = self.__fps_maker(mols_no_lndmks)
        D = self.__fps_distance_to_refps(qfps)

        # Now we can insert the rows in D corresponding to the landmarks
        # We first need to correct the idx_landmarks for the bad_idx positions
        # since these will not have a fingerprint and distance row generated
        for i in bad_idx:
            idx_landmarks[idx_landmarks > i] -= 1

        D_lndmks = self.__refD
        i = 0
        for ix in idx_landmarks:
            D = np.insert(D, ix, D_lndmks[i, :], axis=0)
            i += 1

        return self.__project(D, bad_idx)

    def __project(self, D, bad_idx):

        algorithm = self._algorithm
        LndS = self.__LndS

        if algorithm == "mds":
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

        else:
            X = LndS.transform(D)

        #  We insert nan rows in the bad_idx positions if there were any
        if len(bad_idx)>0:
            ix = [bad_idx[i]-i for i in range(len(bad_idx))]
            X = np.insert(X, ix, np.nan, axis=0)

        return X

    def __lndmk_choice(self, smiles, n_landmarks, Y, Yset, random_state):
        # Returns SORTED indices
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Let us first extract the landmark fingerprints
        # If Y is not provided, pick the landmarks randomly
        lenY = len(Y)
        if lenY == 0:  # Y = []
            idx_landmarks = np.random.choice(
                range(len(smiles)), n_landmarks, replace=False
            )

        else:
            if Yset == "highest":  # gets the landmarks from the top
                idx_landmarks = np.argpartition(Y, -n_landmarks)[-n_landmarks:]
            elif Yset == "lowest":  # get the landmarks from the bottom
                idx_landmarks = np.argpartition(Y, n_landmarks)[:n_landmarks]
            else:  # Yset is an integer
                try:
                    nbins = min(Yset, n_landmarks)
                except TypeError:
                    raise IOError("Invalid Yset argument.")
                Y_binned = np.floor(
                    (Y - min(Y)) / (1.00001 * (max(Y) - min(Y))) * nbins
                )

                proba = np.zeros(lenY)
                for i in range(nbins):
                    idx = np.where(Y_binned == i)[0]
                    if len(idx) > 0:
                        proba[idx] = 1.0 / (len(idx))
                proba = proba / sum(proba)

                idx_landmarks = np.random.choice(
                    range(lenY), n_landmarks, replace=False, p=proba
                )

        return np.sort(idx_landmarks)

    def __refps_update(self):
        remols = self.__remols
        refps = self.__fps_maker(remols)
        self.__refps = refps

    def __refD_update(self):
        refps = self.__refps
        D = self.__self_fps_distance(refps)
        self.__refD = D

    def __safe_lndmk_embed(self):
        # This function tries to call lndmk_embed and if it fails will reduce
        # n_components to make it work.
        copy_n_components = self._n_components
        n_components = self._n_components

        if n_components > self._n_landmarks:
            n_components = self._n_landmarks

        for i in range(n_components, 0, -1):  # will try down to n_components=1
            try:
                self._n_components = i
                self.__lndmk_embed()
                break
            except ValueError:
                pass

        if self._n_components < copy_n_components:
            print(
                f"Not enough Nldnmk, n_components has been set to "
                f"{self._n_components}"
            )

    def __lndmk_embed(self):

        D = self.__refD
        algorithm = self._algorithm
        n_components = self._n_components

        n = len(D)

        if algorithm == "mds":
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
            if n_components:
                arr = evals
                w = arr.argsort()[-n_components:][::-1]
                if np.any(evals[w] < 0):
                    raise ValueError(
                        "Not enough positive eigenvalues for "
                        "the selected n_components."
                    )

            if w.size == 0:
                raise ValueError("Matrix is negative definite.")

            V = evecs[:, w]
            Lh = V.dot(np.diag(1.0 / np.sqrt(evals[w]))).T
            means = np.mean(D, axis=1)
            LndS = (Lh, means)

        elif algorithm == "isomap":
            n_neighbors = int(np.ceil(1.4 * n_components))
            LndS = Isomap(
                n_neighbors=n_neighbors, n_components=n_components, metric="precomputed"
            )
            LndS.fit(D)

        elif algorithm == "pca":
            LndS = PCA(n_components=n_components)
            LndS.fit(D)

        elif algorithm == "ica":
            LndS = FastICA(
                n_components=n_components,
                algorithm="parallel",
                whiten=True,
                max_iter=1000,
                tol=1e-4,
            )
            LndS.fit(D)

        elif algorithm == "fa":
            LndS = FactorAnalysis(n_components=n_components)
            LndS.fit(D)

        self.__LndS = LndS

    # Helper methods
    def _smiles_to_mols(self, smiles, bad_idx=False):
        mols = []
        bad_ind = []
        for i, smi in enumerate(smiles):
            m = MolFromSmiles(
                smi, sanitize=True
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
        type_fp = self._type_fp
               
        switcher = {}
        switcher["morgan2"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)
        switcher["morgan3"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, radius=3, nBits=1024)
        switcher["rdk5"] = lambda m: Chem.RDKFingerprint(m, minPath=1, maxPath=5, fpSize=1024)
        switcher["rdk7"] = lambda m: Chem.RDKFingerprint(m, minPath=1, maxPath=7, fpSize=2048)
        switcher["featmorgan2"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m,
            radius=2, useFeatures=True, useChirality=True, nBits=2048)
        switcher["featmorgan3"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m,
            radius=3, useFeatures=True, useChirality=True, nBits=2048)
        switcher["maccs"] = rdMolDescriptors.GetMACCSKeysFingerprint
        switcher["avalon"] = pyAvalonTools.GetAvalonFP
        switcher["atompairs"] = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect
        switcher["torsion"] = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect
        switcher["pattern"] = lambda m: Chem.PatternFingerprint(m, fpSize=4096)
        switcher["secfp6"] = lambda m: rdMHFPFingerprint.MHFPEncoder(0,0).EncodeSECFPMol(m, radius=3)
        switcher["layered"] = LayeredFingerprint

        type_fps = type_fp.split("+")

        if len(type_fps) > 1:

            def F(mol, type_fps):
                n = 0
                keys = []
                for type_fp_i in type_fps:
                    f = switcher[type_fp_i]
                    fpi = f(mol)
                    for i in fpi.GetOnBits():
                        keys.append(i + n)

                    n += fpi.GetNumBits()

                fp = Chem.DataStructs.ExplicitBitVect(n)
                for i in keys:
                    fp[i] = 1

                return fp

            fps = [F(mol, type_fps) for mol in mols]

        else:  # len(type_fps)==1
            f = switcher[type_fps[0]]
            fps = list(map(f, mols))
            # Equivalent to fps = [f(mol) for mol in mols]

        return fps

    def __self_fps_distance(self, fps):
        metric = self.__metric_to_class()
        l2 = len(fps)
        S = np.zeros((l2, l2))
        np.fill_diagonal(S, 1)
        for i in range(l2 - 1):
            for j in range(i + 1, l2):
                s = DataStructs.FingerprintSimilarity(fps[i], fps[j], metric=metric)
                S[i, j] = s
                S[j, i] = s

        D = 1.0 - S
        return D

    def __fps_distance_to_refps(self, fps1):
        metric = self.__metric_to_class()
        fps2 = self.__refps

        # Let us now compute the matrix of pairwise distances
        l1 = len(fps1)
        l2 = len(fps2)

        S = np.zeros((l1, l2))
        for i in range(l1):
            for j in range(l2):
                s = DataStructs.FingerprintSimilarity(fps1[i], fps2[j], metric=metric)
                S[i, j] = s

        D = 1.0 - S
        return D

    def __metric_to_class(self):
        metric = self._metric

        switcher = {}
        switcher["tanimoto"] = DataStructs.OnBitSimilarity
        switcher["cosine"] = DataStructs.CosineSimilarity
        switcher["dice"] = DataStructs.DiceSimilarity
        # elif metric =='russel': r = DataStructs.RusselSimilarity # non-commutative
        switcher["sokal"] = DataStructs.SokalSimilarity
        switcher["kulczynski"] = DataStructs.KulczynskiSimilarity
        switcher["mcconnaughey"] = DataStructs.McConnaugheySimilarity
        # elif metric == 'tversky': r = lambda x, y: DataStructs.TverskySimilarity(x, y, 0.01, 0.99) # non-commutative
        switcher["braun-blanquet"] = DataStructs.BraunBlanquetSimilarity
        switcher["rogot-goldberg"] = DataStructs.RogotGoldbergSimilarity
        switcher["asymmetric"] = DataStructs.AsymmetricSimilarity
        switcher["manhattan"] = DataStructs.AllBitSimilarity     
        
        r = switcher.get(metric)
        if r is None:
            print(
                f"Warning: Invalid similarity metric {metric}. \
                    metric has been set to Tanimoto."
            )
            self._metric = "tanimoto"
            r = DataStructs.TanimotoSimilarity
            
        return r


# ----- End of Macaw class -----


def Macaw_optimus(
    smiles,
    y,
    fast=True,
    C=20.0,
    problem="auto",
    verbose=False,
    n_components=15,
    algorithm="MDS",
    **kwargs,
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
        Prints intermediate scores for the different `type_fp` and `metric`
        combinations.

    **kwargs : optional
        Allows to pass additional parameters to the Macaw class (other than
        `type_fp`and `metric`).


    Returns
    -------
    Macaw
        Macaw object with the optimal settings identified.


    """
    smiles = list(smiles)
    y = np.array(y)

    leny = len(y)
    if leny != len(smiles):
        raise IOError("`len(smiles)` = {len(smiles)} does not match `len(y)` = {leny}")

    if problem == "auto":
        tmp = set()
        tmp = tmp.union(y[0:100])
        if len(tmp) > 2:
            problem = "regression"
        else:
            problem = "classification"
        print(f"Problem type identified as {problem}")

    # If not specified, we will use the same Y argument for the individual
    # Macaw calls as the Macaw_optimus y argument.
    # In the case of classification, this will amount to using a 'balanced'
    # number of landmarks. I could set `Yset`equal to 2, but it is not necessary.
    if "Y" not in kwargs:
        kwargs["Y"] = y.copy()

    else:  # just checking that if Y is specified it matches len(smiles)
        if (len(kwargs["Y"]) != 0) and (len(smiles) != len(kwargs["Y"])):
            raise IOError(
                f"len(smiles) = {leny} does not match " f"len(Y) = {len(kwargs['Y'])}"
            )

    type_fps = [
        "morgan2",
        "morgan3",
        "rdk5",
        "rdk7",
        "featmorgan2",
        "featmorgan3",
        "maccs",
        "avalon",
        "atompairs",
        "torsion",
        "pattern",
        "secfp6",
        "layered",
    ]

    metrics = [
        "tanimoto",
        "cosine",
        "dice",
        "sokal",
        "kulczynski",
        "mcconnaughey",
        "braun-blanquet",
        "rogot-goldberg",
        "asymmetric",
        "manhattan",
    ]

    mcw = Macaw(n_components=n_components, algorithm=algorithm)
    mcw.fit(smiles, **kwargs)  # Landmark selection

    if problem == "regression":
        epsilon = np.ptp(y) / 25.0
        f = SVR(kernel="rbf", C=C, epsilon=epsilon, verbose=False)
    else:
        f = SVC(kernel="rbf", C=C, gamma="scale", verbose=False)

    M = np.zeros((len(type_fps), len(metrics)))
    if fast:
        cv = 3
        metrics_short = ["tanimoto"]

        J = [metrics.index(m) for m in metrics_short]

        smiles_subset = smiles
        y_subset = y

        if leny > 400:
            np.random.seed(123)
            idx = np.random.choice(range(len(smiles)), 400, replace=False)

            smiles_subset = [smiles[i] for i in idx]
            # Equivalent to list(itemgetter(*idx)(smiles))

            y_subset = y[idx]

        for i, type_fp in enumerate(type_fps):
            mcw.set_type_fp(type_fps[i])

            for j in J:

                if (i, j) == (9, 3):  # torsion + sokal returns nan
                    continue

                mcw.set_metric(metrics[j])

                x = mcw.transform(smiles_subset)

                # splitters are instantiated with shuffle=False so the splits
                # will be the same across calls.
                M[i, j] = cross_val_score(f, x, y_subset, cv=cv, verbose=0).mean()

                if verbose:
                    print(f"{type_fps[i]} + {metrics[j]}: {M[i,j]:0.3f}")

        # Now we select a couple promising fps to evaluate all the
        # similarity metrics on them
        tmp_ranks = M[:, J].argsort(axis=0).argsort(axis=0) + 1
        tmp_ranks = np.sum(tmp_ranks, axis=1)
        # n = len(metrics_short)
        n = 3
        ind = np.argpartition(tmp_ranks, -n)[-n:]

        for i in ind:
            mcw.set_type_fp(type_fps[i])

            for j, metric in enumerate(metrics):
                if M[i, j] != 0:
                    continue
                if (i, j) == (9, 3):  # torsion + sokal returns nan
                    continue

                mcw.set_metric(metric)

                x = mcw.transform(smiles_subset)

                # splitters are instantiated with shuffle=False so the splits
                # will be the same across calls.
                M[i, j] = cross_val_score(f, x, y_subset, cv=cv, verbose=0).mean()

                if verbose:
                    print(f"{type_fps[i]} + {metrics[j]}: {M[i,j]:0.3f}")

    else:
        cv = 5

        for i, type_fp in enumerate(type_fps):
            mcw.set_type_fp(type_fp)

            for j, metric in enumerate(metrics):
                mcw.set_metric(metric)

                x = mcw.transform(smiles)

                # splitters are instantiated with shuffle=False so the splits
                # will be the same across calls.
                M[i, j] = cross_val_score(f, x, y, cv=cv).mean()

                if verbose:
                    print(f"{type_fps[i]} + {metrics[j]}: {M[i,j]:0.3f}")

    # Torsion + sokal combination returns nan's. We set its score to 0.0
    np.nan_to_num(M, copy=False)
    max_i, max_j = np.unravel_index(M.argmax(), M.shape)

    print(f"Setting recommended combination: {type_fps[max_i]} + {metrics[max_j]}")

    # Now we set the embedder to the optimal combination
    mcw.set_type_fp(type_fps[max_i])
    mcw.set_metric(metric=metrics[max_j])

    return mcw
