# MACAW

MACAW (Molecular AutoenCoding Auto-Workaround) is a cheminformatic toolkit that embeds in a low-dimensional, continuous numeric space. The embeddings are molecular features that can be used as inputs in mathematical and machine-learning models.

MACAW embeddings can be used as an alternative for conventional molecular descriptors. MACAW embeddings are fast and easy to compute, variable selection is not needed, and they may enable more accuracte predictive models than conventional molecular descriptors.

MACAW also provides original algorithms to generate molecular libraries and to evolve molecules *in silico* that satisfy a desired specification (inverse molecular design). The design specification can be any property or combination of properties that can be predicted for the molecule, such as its octane number or its binding affinity to a protein.

Details about the different algorithms are explained in the [MACAW publication]().



## Installation

MACAW requires rdkit 2020.09.4 or later to run, which can be installed using [conda](https://anaconda.org/conda-forge/rdkit):

```bash
conda install -c conda-forge rdkit
```

Alternative methods to install rdkit can be found [here](https://www.rdkit.org/docs/Install.html).


Then run the following command to install MACAW:

```bash
pip install macaw_py
```

## Use

The following illustrates some of the main commands in MACAW. Detailed use examples with real datasets are available as Jupyter Notebooks in the [MACAW repository](https://github.com/LBLQMM/macaw).


### Molecule embedding


![MACAW embedder](/results/Figure_readme1.png?raw=true)


Given a list of molecules represented as SMILES strings (`smiles`), their MACAW embeddings (`X`) can be obtained as follows:

```python
from macaw import *

mcw = MACAW()
mcw.fit(smiles)
X = mcw.transform(smiles)
```

Any list of molecules in SMILES format (`newsmiles`) can be embedded using an existing MACAW object:

```python
X_new = mcw.transform(newsmiles)
```

The embedder has a variety of parameters that can be tuned to improve results. These include the dimensionality of the embedding (`n_components`), the number of landmarks used (`n_landmarks`), the type of molecular fingeprint (`type_fp`), and the similarity metric (`metric`). Property values (`y_values`) can also be provided to the argument `Y` to improve landmark choice. The arguments and options available are listed in the class help.

```python
mcw = MACAW(n_components=20, type_fp='rdk5', metric='Dice', n_landmarks=60)

mcw.fit_transform(smiles, Y=y_values)
```

The function `MACAW_optimus` automatically explores a variety of fingeprint type (`type_fp`) and similarity metric (`metric`) combinations and returns a recommended embedder ready to use:

```python
mcw = MACAW_optimus(smiles, n_components=20, y=y_values, verbose=True)
```

### Molecule generation

Given an input dataset of molecules in SELFIES format, MACAW's `library_maker` function will generate a library of molecules around it. The maximum number of molecules to generate is specified with the `n_gen` parameter, while the spread of the distribution can be controlled with the `noise_factor` argument. Additional parameters are explained in the function help.


```python
smiles_lib = library_maker(smiles, n_gen=50000, noise_factor=0.3)
```

### Molecule recommendation (inverse design)


![MACAW evolver](/results/Figure_readme2.png?raw=true)


Given a property of interest, a model `f` can be trained to predict the property values of different molecules. The model `f` takes as inputs the features generated by the embedder `mcw`.

Then, we can evolve and recommend molecules to satisfy a desired property specification value (`spec`) using the function `library_evolver`. It takes as input an initial set of molecules (`smiles`), the featurizer (`mcw`), the predictive model (`f`), the desired specification value (`spec`), the number of molecules ro recommend (`n_hits`), the number of evolution rounds (`n_rounds`). Other optional arguments described in the function help.

```python
recommended_smiles = library_evolver(smiles, mcw, f, spec, n_hits=10, n_rounds=8)
```

## License

MACAW code is distributed under the license specified in the [`Noncomercial_Academic_LA.pdf`](https://github.com/LBLQMM/macaw/Noncomercial_Academic_LA.pdf) file. This license allows free **non-commercial** use for **academic institutions**. Modifications should be fed back to the original repository to benefit all users. 

A separate **commercial** use license is available from [Berkeley Lab](mailto:ipo@lbl.gov). The license terms (10 years) are $5,000 for small businesses (less than 250 employees) and $15,000 for large businesses (more than 250 employees).

An **evaluation license** for commercial users can be obtained for 30 days of testing by filling the [`Evaluation_LA.pdf`](https://github.com/JBEI/ART/blob/master/Evaluation_LA.pdf) file and sending back to [Jean Haemmerle, LBNL Licensing Associate](mailto:jhaemmerle@lbl.gov).
