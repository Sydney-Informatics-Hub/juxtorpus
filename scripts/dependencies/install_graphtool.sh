#!/bin/bash
# This script installs graph tool.
# It is intended to be optional until user needs to use TopSBM.
#
# graph-tools: https://graph-tool.skewed.de/
# TopSBM: https://topsbm.github.io/


2>/dev/null which conda
[[ $? == 0 ]] || 2>& echo "Missing 'conda' dependency. Unable to install graph-tool."

echo "Installing graph-tool..."
conda install -c conda-forge graph_tool -y --quiet

if [[ $? == 0 ]]; then
  echo "Succesfully installed graph-tool."
  exit 0
else
  2>& echo "Failed to install graph-tool. Trying again but verbose."
  conda install -c conda-forge graph_tool -y
fi