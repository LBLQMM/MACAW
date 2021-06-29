from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      name="macaw",
      version='0.0.1',
      url="https://github.com/LBLQMM/macaw",
      author="Vincent Blay",
      author_email="vblayroger@lbl.gov",
      description='Macaw molecular embedder',
      long_description=long_description,
      long_description_content_type="text/markdown",
      py_modules=["MolEmbedding","MolGeneration","Plotting"],
      package_dir={'': 'src'},
      classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research/Development",
        "Intended Audience :: Scientific Engineering",
        "Intended Audience :: Application",
        "Natural Language :: English"
          ],
      install_requires=[
          "numpy",
          "pandas",
          "matplotlib",
          "sklearn >= 0.24.1",
          "scipy >= 1.6.2",
          "selfies >= 1.0.4",
          "rdkit >= 2020.03.6"
          ]
      )

