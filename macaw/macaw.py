# -*- coding: utf-8 -*-
"""
Part of the MACAW project.
Contains the MACAW class, the MACAW_optimus function, and the smiles_cleaner
function.

@author: Vincent Blay, 2021
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMHFPFingerprint
from rdkit.Chem.rdmolops import LayeredFingerprint
import re
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_score
import umap


class MACAW:
    """
    Class providing MACAW numeric embeddings of molecules.

    :param type_fp: Desired type of fingerprint to use to characterize molecules.
        Options include 'RDK5', 'RDK7', 'Morgan2', 'Morgan3',
        'featMorgan2', 'featMorgan3,'Avalon','MACCS', 'atompairs',
        'torsion', 'pattern', 'secfp6', and 'layered'. Combinations can
        also be specified with the '+' symbol, e.g. 'RDK5+MACCS'.
        Defaults to 'Morgan2'.
    :type type_fp: str, optional
    :param metric: str, optional
        Distrance metric used to measure similarity between molecular
        fingerprints. Options include 'Tanimoto', 'dice', 'cosine',
        'Sokal', 'Kulczynski', 'Mcconnaughey', 'Braun-Blanquet',
        'Rogot-Goldberg', 'asymmetric', 'Manhattan', and 'Blay-Roger'.
        Defaults to 'Tanimoto'.
    :type metric: str, optional
    :param n_components: Number of dimensions for the embedding. Defaults to 15.
    :type n_components: int, optional
    :param algorithm: Algorithm to use for the projection. Options available
        are 'MDS', 'isomap', 'PCA', ICA', 'FA', and 'umap'. Defaults to 'MDS'.
    :type algorithm: str, optional
    :param n_landmarks: Desired number of landmark molecules to use.
        Defaults to 50.
    :type n_landmarks: int, optional
    :param Yset: Specifies how to use the input in `Y` during fitting, if provided.
        Options include 'highest' and 'lowest'. If an integer is provided,
        it will use uniform sampling of landmarks after splitting the
        molecules in `Yset` bins. Defaults to 10.
    :param idx_landmarks: List indicating the indices of the molecules to be
        used as landmarks.
    :type idx_landmarks: list, optional
    :type Yset: str or int, optional
    :param random_state: Seed to have the same choice of
        landmarks across runs.
    :type random_state: int, optional

    .. note::
        If the `Y` argument is provided during fitting, it will be used in the
        choice of the landmarks. If `Yset` is an integer, then the dataset will
        be split in `Yset` bins according to `Y` and landmarks will be sampled
        from the bins with equal probability. If `Yset` is set to 'highest' or
        'lowest', then the landmarks will be the `n_landmarks` molecules with
        the highest or lowest `Y` values, respectively.

    .. warning::
        No attribute should be modified directly. Instead, setter methods are
        available for the modifiable attributes: `set_type_fp()`,
        `set_metric()`, `set_Y()`, `set_n_components()`, and `set_algorithm()`.
    """

    __version__ = "alpha_11"
    __author__ = "Vincent Blay"

    def __init__(
        self,
        type_fp="Morgan2",
        metric="Tanimoto",
        n_components=15,
        algorithm="MDS",
        n_landmarks=50,
        Yset=10,
        idx_landmarks=None,
        random_state=None
    ):
        """Constructor method"""

        self._n_components = n_components
        self._type_fp = type_fp.lower().replace(" ", "")
        self._metric = metric.lower().replace(" ", "")

        algorithm = algorithm.lower().replace(" ", "")
        if algorithm not in ["mds", "isomap", "pca", "ica", "fa", "umap"]:
            raise IOError(
                (
                    f"Error: Invalid algorithm {algorithm}. See the "
                    "documentation for available algorithms."
                )
            )
        self._algorithm = algorithm

        self._n_landmarks = n_landmarks

        self._idx_landmarks = idx_landmarks
        self._resmiles = []
        self._isfitted = False

        # These attributes need not be visible or modified by the user,
        # we make them private
        self.__remols = []
        self.__refps = []
        self.__refD = []
        self.__LndS = []
        self.__Yset = Yset
        self.__random_state = random_state

    def __repr__(self):
        return (
            f"MACAW_embedder(n_landmarks={self._n_landmarks}, "
            f"n_components={self._n_components}, type_fp={self._type_fp},"
            f" metric={self._metric}, algorithm={self._algorithm})"
        )

    def __len__(self):
        return self._n_components

    # Setter algorithms

    def set_type_fp(self, type_fp):
        """Method to change the `type_fp` used in an existing MACAW object."""
        self._type_fp = type_fp.lower().replace(" ", "")
        if self._isfitted: 
            self.__refps_update()
            self.__refD_update()
            self.__safe_lndmk_embed()

    def set_metric(self, metric):
        """Method to change the `metric` used in an existing MACAW object."""
        self._metric = metric.lower().replace(" ", "")
        # self.__refps_update()
        if self._isfitted:
            self.__refD_update()
            self.__safe_lndmk_embed()

    def set_n_components(self, n_components):
        """Method to change the `n_components` used in an existing MACAW object."""
        self._n_components = n_components
        if self._isfitted:
            self.__safe_lndmk_embed()

    def set_algorithm(self, algorithm):
        """Method to change the `algorithm` used in an existing MACAW object."""
        algorithm = algorithm.lower().replace(" ", "")
        if algorithm not in ["mds", "isomap", "pca", "ica", "fa", "umap"]:
            raise IOError(f"Unknown algorithm {algorithm}.")
        self._algorithm = algorithm

        if self._isfitted:
            self.__safe_lndmk_embed()

    # Main functions for the embedding

    def fit(
        self,
        smiles,
        Y=None
    ):
        """Method to select the landmarks and initialize the MACAW embedding
        space.

        :param smiles: List of molecules given in SMILES format.
        :type smiles: list
        :param Y: List of property values of interest, one for each molecule in
            `smiles`. If provided, it may help choosing a more diverse
            set of landmark molecules.

        """

        smiles = list(smiles)
        n_landmarks = self._n_landmarks

        if n_landmarks > len(smiles):
            print(
                f"Warning: requested n_landmarks={n_landmarks} but only "
                f"{len(smiles)} smiles provided. n_landmarks will be lower."
            )
        
        idx_landmarks = self._idx_landmarks

        if idx_landmarks is None:
            idx_landmarks = self.__lndmk_choice(smiles, Y)

        resmiles = [smiles[i] for i in idx_landmarks]
        remols, bad_idx = self._smiles_to_mols(resmiles, bad_idx=True)

        if len(bad_idx) > 0:
            idx_landmarks = np.setdiff1d(idx_landmarks, bad_idx)
            print(f"n_landmarks has been set to {len(idx_landmarks)}")

        resmiles = [smiles[i] for i in idx_landmarks]

        self._idx_landmarks = np.sort(idx_landmarks)
        self._n_landmarks = len(idx_landmarks)
        self._resmiles = resmiles
        self.__remols = remols
        self.__refps_update()
        self.__refD_update()
        self.__safe_lndmk_embed()
        self._isfitted = True

    def transform(self, qsmiles):
        """Method to embed a list of molecules in an existing MACAW space.

        :param qsmiles: List of query molecules to be embedded given in SMILES
            format.
        :type qsmiles: list

        :return: A 2D array such that each row is the embedding of each `qsmiles`
            molecule.
        :rtype: numpy.ndarray

        .. note::
            If any invalid SMILES is encountered in the input, the corresponding
            row in the output will be filled with nan's.
        """

        if not self._isfitted: 
            raise RuntimeError(
                "MACAW instance not fitted. Call "
                "'fit' with appropriate arguments before using"
                " this embedder."
            )

        qsmiles = list(qsmiles)

        # We will split the query smiles into batches to reduce memory needs
        batch_size = 50000  # max number of molecules per batch

        L = len(qsmiles)
        X = np.zeros((L, self._n_components))

        for lb in range(0, L, batch_size):
            if lb > 0:
                print(lb)
            ub = lb + batch_size
            batch = qsmiles[lb:ub]

            # We compute the fingerprints and distances for the query smiles
            mols, bad_idx = self._smiles_to_mols(batch, bad_idx=True)
            qfps = self.__fps_maker(mols)
            D = self.__fps_distance_to_refps(qfps)

            x_batch = self.__project(D, bad_idx)

            X[lb:ub, :] = x_batch

        return X

    def fit_transform(
        self,
        qsmiles,
        Y=None
    ):
        """Combination of the `fit` and `transform` methods."""

        qsmiles = list(qsmiles)
        self.fit(
            smiles=qsmiles,
            Y=Y
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
        if len(bad_idx) > 0:
            ix = [bad_idx[i] - i for i in range(len(bad_idx))]
            X = np.insert(X, ix, np.nan, axis=0)

        return X

    def __lndmk_choice(self, smiles, Y):
        # Returns SORTED indices

        if self.__random_state is not None:
            np.random.seed(self.__random_state)

        n_landmarks = self._n_landmarks

        # Let us first extract the landmark fingerprints
        # If Y is not provided, pick the landmarks randomly
        if Y is None:
            idx_landmarks = np.random.choice(
                range(len(smiles)), n_landmarks, replace=False
            )

        else:
            lenY = len(Y)
            Yset = self.__Yset

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

        elif algorithm == "umap":
            LndS = umap.UMAP(
                n_components=n_components, metric="precomputed", min_dist=0.3
            )
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
        switcher["morgan2"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
            m, radius=2, nBits=2048
        )
        switcher["morgan3"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
            m, radius=3, nBits=2048
        )
        switcher["rdk5"] = lambda m: Chem.RDKFingerprint(
            m, minPath=1, maxPath=5, fpSize=2048
        )
        switcher["rdk7"] = lambda m: Chem.RDKFingerprint(
            m, minPath=1, maxPath=7, fpSize=2048
        )
        switcher["featmorgan2"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
            m, radius=2, useFeatures=True, useChirality=True, nBits=2048
        )
        switcher["featmorgan3"] = lambda m: AllChem.GetMorganFingerprintAsBitVect(
            m, radius=3, useFeatures=True, useChirality=True, nBits=2048
        )
        switcher["maccs"] = rdMolDescriptors.GetMACCSKeysFingerprint
        switcher["avalon"] = lambda m: pyAvalonTools.GetAvalonFP(m, nBits=2048)
        switcher[
            "atompairs"
        ] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            m, nBits=2048
        )
        switcher[
            "torsion"
        ] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
            m, nBits=2048
        )
        switcher["pattern"] = lambda m: Chem.PatternFingerprint(m, fpSize=2048)
        switcher["secfp6"] = lambda m: rdMHFPFingerprint.MHFPEncoder(
            0, 0
        ).EncodeSECFPMol(m, radius=3, length=2048)
        switcher["layered"] = lambda m: LayeredFingerprint(m, fpSize=2048)

        type_fps = type_fp.split("+")

        f = switcher[type_fps[0]]
        fps = [f(mol) for mol in mols]  # list(map(f, mols))

        # This loop will only run if len(type_fps)>1
        for type_fp_i in type_fps[1:]:
            f = switcher[type_fp_i]
            fps_i = [f(mol) for mol in mols]
            # Adding two ExplicitBitVectors simply appends them
            fps = [fp + fp_i for fp, fp_i in zip(fps, fps_i)]

        return fps

    def __self_fps_distance(self, fps):
        metric = self.__metric_to_class()
        l2 = len(fps)
        S = np.zeros((l2, l2))
        np.fill_diagonal(S, 1)
        for i in range(l2 - 1):
            for j in range(i + 1, l2):
                # s = DataStructs.FingerprintSimilarity(fps[i], fps[j], metric=metric)
                s = metric(fps[i], fps[j])
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
                # s = DataStructs.FingerprintSimilarity(fps1[i], fps2[j], metric=metric)
                s = metric(fps1[i], fps2[j])
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
        switcher["blay-roger"] = self.__BlayRogerSimilarity

        r = switcher.get(metric)
        if r is None:
            print(
                f"Warning: Invalid similarity metric {metric}. "
                "metric has been set to Tanimoto."
            )
            self._metric = "tanimoto"
            r = DataStructs.TanimotoSimilarity

        return r

    def __BlayRogerSimilarity(self, x, y, a=0.5, b=0.5):
        v1 = DataStructs.RusselSimilarity(x, y, False)
        v2 = DataStructs.RusselSimilarity(y, x, False)
        s = (a * v1 + b * v2) / (a + b)
        return s


# ----- End of MACAW class -----


def MACAW_optimus(
    smiles,
    y,
    exhaustiveness=1,
    C=20.0,
    problem="auto",
    verbose=False,
    random_state=None,
    **kwargs,
):
    """
    Function that identifies and recommends a MACAW embedding for a given
    problem. It does so by evaluating the performance of different embeddings
    as inputs to a support vector machine.

    :param smiles: List of molecules in SMILES format.
    :type smiles: list
    :param y: List containing the property of interest for each molecule in
        `smiles`.
    :type y: list or numpy.ndarray
    :param exhaustiveness: int, optional
        Controls how many combinations of fingeprint types and distance
        metrics to explore. If set to 1, it will only explore individual
        fingeprints. If set to 2, it will explore individual fingeprints and
        combinations of two fingeprints. If set to 3, it will explore
        additional metrics and perform a slower cross-validation.
    :type exhaustiveness: int, optional
    :param C: Regularization hyperparameter for the SVM. Defaults to 20.
    :type C: float, optional
    :param problem: Indicates whether it is a 'regression' or 'classification' problem. It
        determines if the model to use is a SVR or SVC. Defaults to 'auto',
        which will try to guess the problem type.
    :type problem: str, optional
    :param verbose: Prints intermediate scores for the different `type_fp` and `metric`
        combinations.
    :type verbose: bool, optional
    :param random_state: Seed to have the same downsampling and choice of 
        landmarks across runs.
    :type random_state: int, optional
    :param kwargs: optional
        Allows to pass additional parameters to the MACAW class constructor 
        (other than `type_fp` and `metric`).

    :return: MACAW object with the best settings identified.
    :rtype: MACAW
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

    # We will use the same Y argument for the individual
    # MACAW calls as the MACAW_optimus y argument.
    # In the case of classification, this will amount to using a 'balanced'
    # number of landmarks. I could set `Yset` equal to 2, but it is not necessary.
    
    mcw = MACAW( **kwargs )
    mcw.fit(smiles, y)  # Landmark selection

    if problem == "regression":
        epsilon = np.ptp(y) / 25.0
        f = SVR(kernel="rbf", C=C, epsilon=epsilon, verbose=False)
    else:
        f = SVC(kernel="rbf", C=C, gamma="scale", verbose=False)

    scores_dict = {}

    type_fps = __type_fp_lister()

    if exhaustiveness < 3:
        cv = 3
        metrics_short = ["tanimoto"]
        maxlen = 400
    else:
        cv = 5
        metrics_short = ["tanimoto", "cosine", "dice"]
        maxlen = 4000

    if random_state is not None:
        np.random.seed(random_state)

    if leny > maxlen:
        idx = np.random.choice(range(len(smiles)), maxlen, replace=False)

        smiles_subset = [smiles[i] for i in idx]
        # Equivalent to list(itemgetter(*idx)(smiles))

        y_subset = y[idx]
    else:
        smiles_subset = smiles
        y_subset = y

    # First retrieve the best single type_fp
    scores_dict = __scores_getter(
        f,
        mcw,
        smiles_subset,
        y_subset,
        type_fps,
        metrics_short,
        verbose=verbose,
        cv=cv,
        scores_dict={},
    )

    max_key = max(scores_dict, key=scores_dict.get)
    max_type_fp = max_key.split(" & ")[0]

    if exhaustiveness >= 2:
        # Second try to append another type_fp to the best single type_fp
        type_fps = __type_fp_lister([max_type_fp])
        scores_dict = __scores_getter(
            f,
            mcw,
            smiles_subset,
            y_subset,
            type_fps,
            metrics_short,
            verbose=verbose,
            cv=cv,
            scores_dict=scores_dict,
        )

        max_key = max(scores_dict, key=scores_dict.get)
        max_type_fp = max_key.split(" & ")[0]

    # Finally, explore all distance metrics
    
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
        "blay-roger",
    ]
    
    scores_dict = __scores_getter(
        f,
        mcw,
        smiles_subset,
        y_subset,
        [max_type_fp],
        metrics,
        verbose=verbose,
        cv=cv,
        scores_dict=scores_dict,
    )

    max_key = max(scores_dict, key=scores_dict.get)
    max_type_fp, max_metric = max_key.split(" & ")

    print(f"Setting recommended combination: {max_key}")

    # Now we set the embedder to the optimal combination
    mcw.set_type_fp(max_type_fp)
    mcw.set_metric(max_metric)

    return mcw


def __type_fp_lister(type_fp=None):

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

    if type_fp is None:
        return type_fps
    else:
        if not isinstance(type_fp, list):
            type_fp = [type_fp]

        list_type_fps = []
        for t in type_fp:
            type_fps_new = np.setdiff1d(type_fps, t)
            for type_fp_new in type_fps_new:
                tmp = t + "+" + type_fp_new
                list_type_fps.append(tmp)
        return list_type_fps


def __scores_getter(
    f, mcw, smiles, y, type_fps, metrics, verbose=False, cv=5, scores_dict={}
):
    if not isinstance(type_fps, list):
        type_fps = [type_fps]
    if not isinstance(metrics, list):
        metrics = [metrics]

    for type_fp in type_fps:

        mcw.set_type_fp(type_fp)

        for m in metrics:

            if ("torsion" in type_fp) & (m == "sokal"):
                continue

            mcw.set_metric(m)

            x = mcw.transform(smiles)

            # splitters are instantiated with shuffle=False so the splits
            # will be the same across calls.
            score = cross_val_score(f, x, y, cv=cv, verbose=0).mean()

            key = type_fp + " & " + m
            scores_dict[key] = score

            if verbose:
                print(f"{key}: {score:0.3f}")

    return scores_dict


def smiles_cleaner(smiles, return_idx=False, deep_clean=False):
    """Function to remove invalid SMILES from a list.

    :param smiles: List of molecules in SMILES format.
    :type smiles: list
    :param return_idx: Specifies whether to return indices or not.
        Defaults to False.
    :type return_idx: bool, optional
    :param deep_clean: Applies certain string replacements to the SMILES
        to improve their compatibility with SELFIES. Defaults to False.
    :type deep_clean: bool, optional

    :return: Returns a list containing only the valid SMILES, in the same order
        as the input.
        If `return_idx` is set to True, the return will be a tuple with three
        lists. The first list contains the valid SMILES,
        the second list contains the indices of the valid SMILES,
        and the third list contains the indices of the invalid SMILES in the
        input.
    :rtype: list or tuple
    
    .. note::
        We recommend to set `deep_clean=True` if preparing an input library
        for the `library_evolver` function.

    """
    smiles = list(smiles)
    idx = []
    idx_bad = []
    clean_smiles = []
    for i, s in enumerate(smiles):
        try:
            if deep_clean:
                smi = s.replace(" ", "")
                smi = smi.replace(".", "")
                # This deals with SMILES atoms in brackets, like [C@H]
                # The only exceptions allowed are for tokens of the nitro group
                # which are robust in SELFIES 2.0
                for m in re.findall("\[.*?\]", smi):
                    if m not in ['[N+]', '[O-]']:
                        smi = smi.replace(m, m[1].upper())
                smi = smi.replace("/C", "C")
                smi = smi.replace("\\C", "C")
                smi = smi.replace("/c", "c")
                smi = smi.replace("\\c", "c")
                
            else:
                smi = s
            
            m = Chem.MolFromSmiles(smi, sanitize=True)
        except:
            m = None

        if m is not None:
            idx.append(i)
            clean_smiles.append(smi)
        else:
            idx_bad.append(i)
            print(f"Warning: invalid SMILES in position {i}: {s}")

    if return_idx:
        return clean_smiles, idx, idx_bad
    else:
        return clean_smiles
