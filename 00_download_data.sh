#!/bin/bash
mkdir -p data

if [ ! -f data/urban-dictionary-words-dataset.zip ]; then
  echo "Downloading data..."
  curl -L -o data/urban-dictionary-words-dataset.zip \
    https://www.kaggle.com/api/v1/datasets/download/therohk/urban-dictionary-words-dataset
else
  echo "Data already downloaded"
fi

if [ ! -f data/urbandict-word-defs.csv ]; then
  echo "Unzipping data..."
  unzip data/urban-dictionary-words-dataset.zip -d data/
else
  echo "Data already unzipped"
fi

echo "First 5 lines:"
head -n 5 data/urbandict-word-defs.csv

echo "Total lines:"
wc -l data/urbandict-word-defs.csv

echo "Done"
