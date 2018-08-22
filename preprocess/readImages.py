from os import listdir
from os.path import isfile, join, isdir

import numpy as np
import preprocess.createHashtagsFile as createHashtagsFile
import tensorflow as tf

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS

def read_hash_tags(hash_tag_filepath, hashtag_id_lookup):
    hash_tags = []
    with open(hash_tag_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") and ' ' not in line:
                hashtag = line[1:].lower()
                if hashtag and hashtag in hashtag_id_lookup:
                    hash_tags.append(hashtag_id_lookup[hashtag])
    # TODO: Remove duplicates
    return sorted(hash_tags)


def get_fixed_size_list(hashtags):
    if len(hashtags) >= FLAGS.labels_per_image:
        return hashtags[:FLAGS.labels_per_image]
    if len(hashtags) < FLAGS.labels_per_image:
        hashtags.extend([0] * (FLAGS.labels_per_image - len(hashtags)))
        return hashtags


def create_encoded_hashtags(hashtags):
    ret = np.zeros([FLAGS.label_set_size], dtype=int)
    for hashtag in hashtags:
        ret[hashtag] = 1
    return ret.tolist()


def read_image_and_tags(dir_path, image_and_tags):
    hashtag_id_lookup = createHashtagsFile.get_hashtag_label_set()
    files = listdir(dir_path)
    for file in files:
        file_path = join(dir_path, file)
        if file.endswith(".jpg"):
            hash_tag_filepath = join(dir_path, file[0:-4] + ".txt")
            if isfile(hash_tag_filepath):
                hashtags = read_hash_tags(hash_tag_filepath, hashtag_id_lookup)
                if hashtags:
                    hashtags = create_encoded_hashtags(hashtags)
                    image_and_tags.append((file_path, hashtags))


def read_all_directories(root_path):
    image_and_tags = []
    files = listdir(root_path)
    for file in files:
        file_path = join(root_path, file)
        if isdir(file_path):
            read_image_and_tags(file_path, image_and_tags)
    return image_and_tags
