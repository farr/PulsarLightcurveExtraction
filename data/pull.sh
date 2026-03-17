#!/bin/zsh

set -e

rsync -e ssh -avz --progress sfgateway:/mnt/home/wfarr/PulsarLightcurveExtraction/data/ ./