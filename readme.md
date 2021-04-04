# Humans use learning progress in curiosity-driven exploration
Here you can find all of the custom code used to obtain the results for the study entitled **"Humans use learning progress in curiosity-driven exploration"**.

The article preprint is published on [PsyArxiv](https://psyarxiv.com/7dbr6/) and raw data are available online at the [OSF website](https://osf.io/k2yur/) as well as here. The manuscript is under revision in Nature Communications (we will post a link once the paper gets published).

Statistical analyses and data visualizations were done in [Python](https://www.python.org/) 3.6 and [R](https://www.r-project.org/) 3.5.0. Python and R libraries are listed, respectively, in `py_requirements.txt` and `R_packages.txt` files. Note that Python requirements was automatically generated using `pip freeze`, so it probably contains more than is really essential. Python code mainly relies on `matplotlib`, `seaborn`, `pandas`, `numpy`, `scipy`; and somewhat less on `IPython`, `statsmodels`, `tqdm`, `PIL`, and `numdifftools`. However, to run all notebooks, all of these libraries are needed.
## Jupyter notebooks
Scripts (both Python and R) that were used to produce the figures for the article are named accordingly, and their outputs are saved in the "figures" folder. For convenience, these scripts were integrated in the corresponding [jupyter notebooks](https://jupyter.org/):
- **prepare_data.ipynb** contains functions for manipulating and processing various datasets used for different analyses. These functions can be used to build up datasets used for producing the figures in `figures.ipynb`, fitting models in `computational_model.ipynb`, and running the analyses in `analyses.ipynb`.
- **figures.ipynb** contains functions used to produce **all** of the figures featured in the main manuscript as well as the supplementary information.
- **computational_model.ipynb** contains code for fitting the computational model of choice data (see the article for details)
- **analyses.ipynb** contains code for running all numberical analyses reported in the manuscript and supplementary information.

*All code is written by Alexandr Ten*
