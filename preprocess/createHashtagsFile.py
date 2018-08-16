import argparse
from os import listdir
from os.path import isfile, join, isdir
from operator import itemgetter


def read_hashtags(file, hashtags):
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") and ' ' not in line:
                hashtag = line[1:].lower()
                if not hashtag:
                    return
                if hashtag in hashtags:
                    hashtags[hashtag] += 1
                else:
                    hashtags[hashtag] = 1

def process_directory(dir, hashtags):
    files = listdir(dir)
    for file in files:
        filePath = join(dir, file)
        if isfile(filePath):
            if file.endswith(".txt"):
                read_hashtags(filePath, hashtags)
        elif isdir(filePath) and file != "extra":
            process_directory(filePath, hashtags)

def writeToFile(hashtags):
    with open(FLAGS.frequency_output, 'w') as f:
        index = 0
        for key, value in sorted(hashtags.items(), key=itemgetter(1), reverse=True):
            f.write('%6s %s: %s\n' % (index, key, value))
            index += 1

    with open(FLAGS.index_map_output, 'w') as f:
        index = 0
        for key, value in sorted(hashtags.items(), key=itemgetter(1), reverse=True):
            f.write('%6s %s\n' % (index, key))
            index += 1

def main():
    hashtags = {}
    process_directory(FLAGS.dataset_dir, hashtags)
    writeToFile(hashtags)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--dataset_dir',
      type=str,
      default='/Users/namitr/tfprograms/dataset',
      help='Absolute path to dataset.'
    )
    parser.add_argument(
      '--frequency_output',
      type=str,
      default='/Users/namitr/tfprograms/dataset/hashtag_frequency.txt',
      help='Frequency of hashtags.'
    )
    parser.add_argument(
      '--index_map_output',
      type=str,
      default='/Users/namitr/tfprograms/dataset/hashtag_index_map.txt',
      help='Frequency of hashtags.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()