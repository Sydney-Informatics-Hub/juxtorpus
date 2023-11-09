#!/bin/zsh

set -e

HELP="./$(basename $0) <venv-name>"

if [[ -z $1 ]]; then
  echo $HELP >&2
  exit 1;
fi

VENV_DIR=$1
POETRY_FILE="./pyproject.toml"
REQ_FILE="./requirements.dev.txt"

echo "++ Initialising dev environment..."
if [[ -d $VENV_DIR ]]; then
  printf "-- Virtual environment $VENV_DIR already exists. Replace(y/n)? "
  read x
  [[ $x != 'y' ]] && echo "Exited." && exit 0
  rm -rf $VENV_DIR
fi

echo "++ Creating virtual env at $VENV_DIR..."
python3 -m venv $VENV_DIR

echo "++ Activating virtual env..."
source $VENV_DIR/bin/activate

echo "++ Installing dependencies..."
if [[ -f $POETRY_FILE ]]; then
  echo "++ Found poetry config file."
  pip install --upgrade pip
  pip install poetry
  poetry install --all-extras
elif [[ -f $REQ_FILE ]]; then
  echo "++ Poetry config file not found. Found requirements.txt."
  set +e
  pip install --upgrade pip
  pip install -r $REQ_FILE
else
  echo "-- Neither $POETRY_FILE or $REQ_FILE is found. No dependencies installed."
  exit 1
fi;

echo "++ Done. Your virtual env is installed at $VENV_DIR"
echo "To activate your virtual env run: source $VENV_DIR/bin/activate"
exit 0
