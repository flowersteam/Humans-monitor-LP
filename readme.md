# Code repository
Here you can find all of the custom code used to obtain the results for the study entitled **"Humans monitor learning progress in curiosity-drived exploration"**

Article preprint is published on [PsyArxiv](https://psyarxiv.com/7dbr6/)  and raw data are available online at the [OSF website](https://osf.io/k2yur/) as well as here.

## Description of the files
Statistical analyses and data visualizations were done in [Python](https://www.python.org/) 3.6 and [R](https://www.r-project.org/) 3.5.0. Python dependencies are listed in the "py_requirements.txt" file, and the required packages for R are given as plain text in the "R_packages.txt".

Scripts (both Python and R) that were used to produce the figures for the article are named accordingly, and their outputs are saved in the "figures" folder. For convenience, these scripts were integrated in the corresponding [jupyter notebooks](https://jupyter.org/):
- **prepare_data.ipynb** contains functions for manipulating and processing various datasets used for different analyses.
- **prepare_figures.ipynb** contains function for generating most of the figures included in the article.
- **computational_model.ipynb** contains code for fitting the computational model of choice data (see the article for details)
