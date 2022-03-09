###################################
Welcome to the MACAW documentation!
###################################

**MACAW** (Molecular AutoenCoding Auto-Workaround) is a cheminformatic tool for Python that embeds molecules in a low-dimensional, continuous numeric space. The embeddings are molecular features that can be used as inputs in mathematical and machine-learning models.

MACAW embeddings can be used as an alternative to conventional molecular descriptors. MACAW embeddings are fast and easy to compute, variable selection is not needed, and they may enable more accuracte predictive models than conventional molecular descriptors.

MACAW also provides original algorithms to generate molecular libraries and to evolve molecules *in silico* to meet a desired specification (inverse molecular design). The design specification can be any property or combination of properties that can be predicted for the molecule, such as its octane number or its binding affinity to a protein. Details about the algorithms can be found in the `MACAW publication <https://doi.org/10.26434/chemrxiv-2022-x647j>`_.

.. contents::
    :depth: 3

************
Installation
************

MACAW requires rdkit 2020.09.4 or later to run, which can be installed using 
`conda <https://anaconda.org/conda-forge/rdkit>`_:

.. code-block:: bash

    conda install -c conda-forge rdkit

Alternative methods to install rdkit are given `here <https://www.rdkit.org/docs/Install.html>`_.

.. warning:: rdkit has to be installed manually and is not automatically installed by pip as a dependency.


Then run the following command to install MACAW:

.. code-block:: bash

    pip install macaw_py

*****
Usage
*****

The different MACAW functions can be imported in Python using:

.. code-block:: python

    from macaw import *


Molecule embedding
==================

.. autoclass:: macaw.MACAW
    :members: fit, transform, fit_transform, set_type_fp, set_metric, set_n_components, set_algorithm

.. autofunction:: macaw.MACAW_optimus


Molecule generation
===================

.. autofunction:: generators.library_maker


On-specification molecule evolution
===================================
 
.. autofunction:: generators.library_evolver


Other functions
===============

.. autofunction:: generators.hit_finder

.. autofunction:: generators.hit_finder2

.. autofunction:: macaw.smiles_cleaner


************
How to cite?
************

.. code-block:: bib

    @article{doi:10.26434/chemrxiv-2022-x647j,
    author = {Blay, Vincent and Radivojevich, Tijana and Allen, Jonathan E. and Hudson, Corey M. and Garcia-Martin, Hector},
    title = {MACAW: an accessible tool for molecular embedding and inverse molecular design},
    journal = {ChemRxiv},
    volume = {0},
    number = {ja},
    pages = {null},
    year = {2022},
    URL = {https://doi.org/10.26434/chemrxiv-2022-x647j},
    eprint = {https://doi.org/10.26434/chemrxiv-2022-x647j}
    }

