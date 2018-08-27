from os import listdir
from os.path import isfile, join, isdir

import preprocess.createHashtagsFile as createHashtagsFile
import tensorflow as tf

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS

def _read_hash_tags(hash_tag_filepath, hashtag_id_lookup):
    hash_tags = []
    with open(hash_tag_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") and ' ' not in line:
                hashtag = line[1:].lower()
                if hashtag and hashtag in hashtag_id_lookup:
                    hash_tags.append(hashtag_id_lookup[hashtag])
    return sorted(list(set(hash_tags)))


def _read_image_and_tags(file_path, image_and_tags, hashtag_id_lookup):
    if file_path.endswith(".jpg"):
        hash_tag_filepath = file_path[0:-4] + ".txt"
        if isfile(hash_tag_filepath):
            hashtags = _read_hash_tags(hash_tag_filepath, hashtag_id_lookup)
            if hashtags:
                image_and_tags.append((file_path, hashtags))


def _read_directories_recursive(root_path, image_and_tags, hashtag_id_lookup):
    files = listdir(root_path)
    for file in files:
        file_path = join(root_path, file)
        if isdir(file_path):
            _read_directories_recursive(file_path, image_and_tags, hashtag_id_lookup)
        elif isfile(file_path):
            _read_image_and_tags(file_path, image_and_tags, hashtag_id_lookup)


def read_all_directories(root_path):
    image_and_tags = []
    hashtag_id_lookup = createHashtagsFile.get_hashtag_label_set()
    _read_directories_recursive(root_path, image_and_tags, hashtag_id_lookup)
    return image_and_tags
