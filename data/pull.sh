#!/bin/zsh

set -e

rsync -e ssh -avz --progress rusty:/mnt/home/wfarr/PulsarLightcurveExtraction/data/ ./