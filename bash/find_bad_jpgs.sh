#!/bin/bash

files_processed=0

function process_file() {
  if [[ $1 != *"jpg" ]]; then
     return
  fi
  local error=`djpeg -fast -grayscale -onepass $1 2>&1 > /dev/null`
  if [[ ! -z $error ]]; then
     echo $1 ": " $error
     echo "$1" >> ~/Desktop/bad_filenames.txt
  #else
  #   echo "No error $1"
  fi
  files_processed=$((files_processed+1))
  if [[ $((files_processed%500)) == 0 ]]; then
     echo "Processed $files_processed images"
  fi
}

function process_directory() {
  if [[ $1 =~ "extra" ]]; then
     return
  fi
  if [[ $1 =~ "original" ]]; then
     return
  fi
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

echo "" > ~/Desktop/bad_filenames.txt
process_directory ~/Desktop/food-101
