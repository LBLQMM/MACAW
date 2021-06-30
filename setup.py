from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
      name='macaw',
      version='0.0.3',
      url='https://github.com/LBLQMM/macaw',
      author='Vincent Blay',
      author_email='vblayroger@lbl.gov',
      description='Macaw molecular embedder',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['macaw'],
      classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Natural Language :: English'
          ],
      install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'scikit-learn >= 0.24.1',
          'scipy >= 1.6.2',
          'selfies >= 1.0.4',
          ],
      )