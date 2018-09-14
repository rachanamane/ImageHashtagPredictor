#!/bin/bash

files_processed=0

function process_file() {
  if [[ $1 != *"jpg" ]]; then
     return
  fi
  echo "Processing $1"
  python ~/tfprograms/ImageHashtagPredictor/model/predict.py --image_path=$1
}

function process_directory() {
  echo "Processing $1"
  for filename in $1/*; do
    if [[ -d $filename ]]; then
        process_directory "$filename"
    elif [[ -f $filename ]]; then
        process_file $filename
    else
        echo "Bad argument: $1"
    fi
  done
}

process_directory ~/tfprograms/prediction
