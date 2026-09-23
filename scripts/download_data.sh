#!/usr/bin/env bash
# Helper notes for obtaining the public datasets used in this study.
# The datasets are not redistributed with this repository.
set -e

echo "MovieLens 1M / 10M:  https://grouplens.org/datasets/movielens/"
echo "  - download ml-1m.zip and ml-10m.zip, unzip, and point configs/datasets.yaml"
echo "    at the folders that contain 'ratings.dat'."
echo
echo "Amazon Books reviews:  https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/"
echo "  - download the Books 5-core JSON, place it as Books_5.json, and point"
echo "    configs/datasets.yaml 'books.path' at its folder."
