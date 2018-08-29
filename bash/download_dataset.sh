#!/bin/bash

if [ -z "$1" ]
  then
    echo "Provide search hashtag (without the #) as an argument"
    exit 1
fi

DIR=~/tfprograms/dataset
mkdir -p $DIR
cd $DIR

count=100

echo "Downloading $count posts to $DIR/#$1"

instaloader --no-videos --count=$count --no-compress-json --quiet --post-filter="not is_video" --filename-pattern="{owner_id}-{date_utc:%Y-%m-%d-%H-%M-%S}" --geotags "#$1"
find $DIR/#$1 -name *old*txt -exec rm {} \;

echo "Extracting hashtags in .txt files and moving other files to /extra directory"

mkdir -p $DIR/#$1/extra

for filename in $DIR/#$1/*txt; do
  name=${filename:0:${#filename}-4}
  mv $filename ${name}_desc.txt
  grep -o -e "#\\w*" ${name}_desc.txt >> $filename
  mv ${name}_desc.txt $DIR/#$1/extra
  mv ${name}.json $DIR/#$1/extra
done
