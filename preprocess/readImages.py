from os import listdir
from os.path import isfile, join, isdir

import preprocess.createHashtagsFile as createHashtagsFile
import tensorflow as tf

# Unused import - Required for flags - Don't remove
import shared.flags
FLAGS = tf.app.flags.FLAGS

def _read_image_metadata(hash_tag_filepath, hashtag_id_lookup):
    hash_tags = []
    with open(hash_tag_filepath, 'r') as f:
        line = f.readline().strip()
        current_hashtags = line.split(",")
        for current_hashtag in current_hashtags:
            current_hashtag = current_hashtag.lower()
            if current_hashtag and current_hashtag in hashtag_id_lookup:
                hash_tags.append(hashtag_id_lookup[current_hashtag])
        user_id = f.readline().strip()
    return sorted(list(set(hash_tags))), int(user_id)


def _read_image_and_tags(file_path, image_and_tags, user_history, hashtag_id_lookup):
    if file_path.endswith(".jpg"):
        hash_tag_filepath = file_path[0:-4] + ".txt"
        if isfile(hash_tag_filepath):
            hashtags, user_id = _read_image_metadata(hash_tag_filepath, hashtag_id_lookup)
            if hashtags:
                image_and_tags.append((file_path, hashtags, user_id))
                for hashtag in hashtags:
                    user_history[user_id][hashtag] += 1


def _read_directories_recursive(root_path, image_and_tags, user_history, hashtag_id_lookup):
    files = listdir(root_path)
    for file in files:
        file_path = join(root_path, file)
        if isdir(file_path) and file != "extra" and file != "original":
            _read_directories_recursive(file_path, image_and_tags, user_history, hashtag_id_lookup)
        elif isfile(file_path):
            _read_image_and_tags(file_path, image_and_tags, user_history, hashtag_id_lookup)


def _create_empty_user_history():
    user_history = []
    for i in range(10):
        user_history.append([])
        curr_history = user_history[i]
        for j in range(FLAGS.label_set_size):
            curr_history.append(0)
    return user_history




def read_all_directories(root_path):
    image_and_tags = []
    user_history = _create_empty_user_history()
    hashtag_id_lookup = createHashtagsFile.get_hashtag_label_set()
    _read_directories_recursive(root_path, image_and_tags, user_history, hashtag_id_lookup)
    return image_and_tags, user_history
