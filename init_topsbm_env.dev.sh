#!/bin/zsh

if [[ -z $1 ]]; then
  1>&2 echo "./$(basename $0) <new-env-name>"
  exit 1;
fi

set -e
echo "+ This script will stop when an error occurs."

ENV_NAME=$1
conda create -n $ENV_NAME python==3.9.17 -y
# can't use conda activate in scripts unless conda init https://github.com/conda/conda/issues/7980
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Python path: $(which python)"

# install graph-tool
conda install graph-tool -c conda-forge -y
pip install poetry

poetry install

# this is hacky since they don't have poetry or setup.py
# it'll basically clone the topsbm repo into your virtual env.
path_site_packages=$(find $(which python | sed 's/\/bin\/python//') -name 'site-packages' -print -quit)
git clone https://github.com/martingerlach/hSBM_Topicmodel.git "$path_site_packages/topsbm"

echo "+ Now run: conda activate $ENV_NAME && python test_topsbm.py"
echo "The above command should run."
echo "The issue is now, try conda install pytorch -c pytorch"
echo "Then run: python -c 'import torch; import graph_tool.all;'"