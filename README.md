# Juxtorpus

Compare and analyse two corpus.

To run this notebook, click on the button below to launch a free binder instance.

[//]: # ([![Binder]&#40;https://binderhub.atap-binder.cloud.edu.au/badge_logo.svg&#41;]&#40;https://binderhub.atap-binder.cloud.edu.au/v2/gh/Sydney-Informatics-Hub/juxtorpus/DH_workshop_140323?labpath=notebooks%2FDH%20demo%2FDemo-final.ipynb&#41;)

[//]: # ([![Binder]&#40;https://binderhub.atap-binder.cloud.edu.au/badge_logo.svg&#41;]&#40;https://binderhub.atap-binder.cloud.edu.au/v2/gh/Sydney-Informatics-Hub/juxtorpus/arrnet_presentation?labpath=notebooks%2Fdemos%2FDemo-ARRNet.ipynb&#41;)
[![Binder](https://binderhub.atap-binder.cloud.edu.au/badge_logo.svg)](https://binderhub.atap-binder.cloud.edu.au/v2/gh/Sydney-Informatics-Hub/juxtorpus/feat/integration_concordance?labpath=notebooks%2Fdemos%2Fdigital_humanities_day%2Fdh_day.ipynb)

# Installation

Currently this is not published. Install as dependency from github:

```shell
pip install 'juxtorpus @ git+https://github.com/Sydney-Informatics-Hub/juxtorpus.git@<commit>'
# install extras: visualisations
pip install 'juxtorpus[viz] @ git+https://github.com/Sydney-Informatics-Hub/juxtorpus.git@<commit>'
# install extras: visualisations + apple (if you're using a Mac)
pip install 'juxtorpus[viz,apple] @ git+https://github.com/Sydney-Informatics-Hub/juxtorpus.git@<commit>'
```

# Contributors

A quick way to set up your dev environment.

```shell
./init_env.dev.sh
```

To build the `requirements.txt` file for binder:

```shell
./binder_poetry_to_requirements.sh
```