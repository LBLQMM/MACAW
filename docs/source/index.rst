***********************************
Welcome to the MACAW documentation!
***********************************

**MACAW** (Molecular AutoenCoding Auto-Workaround) is a cheminformatic tool for Python that embeds molecules in a low-dimensional, continuous numeric space. The embeddings are molecular features that can be used as inputs in mathematical and machine-learning models.

MACAW embeddings can be used as an alternative for conventional molecular descriptors. MACAW embeddings are fast and easy to compute, variable selection is not needed, and they may enable more accuracte predictive models than conventional molecular descriptors.

MACAW also provides original algorithms to generate molecular libraries and to evolve molecules *in silico* that satisfy a desired specification (inverse molecular design). The design specification can be any property or combination of properties that can be predicted for the molecule, such as its octane number or its binding affinity to a protein.

.. contents::
    :depth: 3

Installation
============

MACAW requires rdkit 2020.09.4 or later to run, which can be installed using [conda](https://anaconda.org/conda-forge/rdkit):

.. code-block:: bash

    conda install -c conda-forge rdkit


.. note:: rdkit has to be installed manually and is not automatically installed by pip as a dependency.


Then run the following command to install MACAW:

.. code-block:: bash

    pip install macaw_py

Usage
=====

Molecule embedding
``````````````````

.. autoclass:: macaw.MACAW

.. autofunction:: macaw.MACAW_optimus


Random molecule generation
``````````````````````````

.. autofunction:: generators.library_maker


On-specification molecule evolution
```````````````````````````````````
 
.. autofunction:: generators.library_evolver


Additional functions
````````````````````

.. autofunction:: generators.hit_finder

.. autofunction:: generators.hit_finder2

.. autofunction:: macaw.smiles_cleaner


How to cite?
============

.. code-block:: bib

    @article{doi:XX.XXX,
    author = {Blay, Vincent and Radivojevich, Tijana and Garcia-Martin, Hector},
    title = {MACAW: XXXX},
    journal = {XXXX},
    volume = {0},
    number = {ja},
    pages = {null},
    year = {0},
    doi = {XX.XXX},

    URL = {http://dx.doi.org/XX.XXXX},
    eprint = {http://dx.doi.org/XX.XXXX}
    }

