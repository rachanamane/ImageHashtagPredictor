from os import listdir
from os.path import isfile, join, isdir
from operator import itemgetter

import tensorflow as tf

# Unused import - Required for flags - Don't remove
import shared.flags

FLAGS = tf.app.flags.FLAGS


def _read_hashtags(file, hashtags):
    with open(file, 'r') as f:
        line = f.readline().strip()
        current_hashtags = line.split(",")
        for hashtag in current_hashtags:
            hashtag = hashtag.lower()
            if not hashtag:
                continue
            if hashtag in hashtags:
                hashtags[hashtag] += 1
            else:
                hashtags[hashtag] = 1


def _process_directory(dir, hashtags):
    files = listdir(dir)
    for file in files:
        filePath = join(dir, file)
        if isfile(filePath):
            if file.endswith(".txt"):
                _read_hashtags(filePath, hashtags)
        elif isdir(filePath) and file != "extra":
            _process_directory(filePath, hashtags)


def _write_hashtag_label_set_file(hashtags):
    hashtag_id_lookup = {}
    with open(FLAGS.hashtags_output_file, 'w') as f:
        index = 0
        f.write("%s\n" % FLAGS.dataset_dir)
        for key in hashtags:
            f.write('%6s %s\n' % (index, key))
            hashtag_id_lookup[key] = index
            index += 1
    return hashtag_id_lookup


def _read_hashtag_label_set_file():
    hashtag_id_lookup = {}
    with open(FLAGS.hashtags_output_file, 'r') as f:
        is_first_line = True
        for line in f:
            if is_first_line:
                is_first_line = False
                continue
            words = line.strip().split(" ")
            hashtag_id_lookup[words[1]] = int(words[0])
    return hashtag_id_lookup


def _create_hashtag_label_set(should_write=False):
    hashtags = {}
    _process_directory(FLAGS.dataset_dir, hashtags)
    hashtags = dict(sorted(hashtags.items(), key=itemgetter(1), reverse=True)[:FLAGS.label_set_size])
    return _write_hashtag_label_set_file(hashtags)


def _get_previous_hashtag_dir():
    with open(FLAGS.hashtags_output_file, 'r') as f:
        for line in f:
            return line.strip()


def get_hashtag_label_set(force_overwrite=False):
    """
    Returns a hashtag lookup dictionary, with key as the hashtag string
    and value is a unique label ID corresponding to the hashtag
    """
    if isfile(FLAGS.hashtags_output_file) and not force_overwrite:
        previous_dir = _get_previous_hashtag_dir()
        if previous_dir == FLAGS.dataset_dir:
            return _read_hashtag_label_set_file()
        else:
            return _create_hashtag_label_set(should_write=True)
    else:
        return _create_hashtag_label_set(should_write=True)


#if __name__ == "__main__":
#    _create_hashtag_label_set(should_write=True)
