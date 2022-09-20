#!/bin/bash

path=pathname/valentini/tt
if [[ ! -e $path ]]; then
    mkdir -p $path
fi
python3 -m model.audio ***/valentini/noisy > $path/noisy.json
python3 -m model.audio ***/valentini/clean > $path/clean.json
