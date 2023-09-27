#!/bin/zsh

set -e

HELP="./$(basename $0) <virutal-env-name>"

if [[ -z $1 ]]; then
  echo "-- $HELP" >&2
  exit 1
fi

VENV_DIR=$1
POETRY_FILE="./pyproject.toml"
OS=$(uname -o)
ARCH=$(uname -m)

if [[ ! -f $POETRY_FILE ]]; then
  echo "-- Missing $POETRY_FILE." >&2
  exit 1
fi

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
echo "++ Found poetry config file."
pip install --upgrade pip
pip install poetry

#:u - uppercase (zsh only)
if [[ ${ARCH:u} == ARM* && ${OS:u} == "DARWIN" ]]; then
  poetry install --with "dev,viz,macOS"
else
  poetry install --with "dev,viz"

echo "++ Done. Your virtual env is installed at $VENV_DIR"
echo "To activate your virtual env run: source $VENV_DIR/bin/activate"
exit 0
