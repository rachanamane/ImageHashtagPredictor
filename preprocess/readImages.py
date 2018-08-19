from os import listdir
from os.path import isfile, join, isdir

def read_hash_tags(hash_tag_filepath):
    # TODO: Return hashtag_index instead of strings
    hash_tags = []
    index = 1
    with open(hash_tag_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") and ' ' not in line:
                hashtag = line[1:].lower()
                if hashtag:
                    # hash_tags.append(hashtag)
                    hash_tags.append(index)
                    index += 1
    # TODO: Remove duplicates
    return hash_tags


def read_image_and_tags(dir_path, image_and_tags):
    files = listdir(dir_path)
    for file in files:
        file_path = join(dir_path, file)
        if file.endswith(".jpg"):
            hash_tag_filepath = join(dir_path, file[0:-4] + ".txt")
            if isfile(hash_tag_filepath):
                hashtags = read_hash_tags(hash_tag_filepath)
                image_and_tags.append((file_path, hashtags))

def read_all_directories(root_path):
    image_and_tags = []
    files = listdir(root_path)
    for file in files:
        file_path = join(root_path, file)
        if isdir(file_path):
            read_image_and_tags(file_path, image_and_tags)
    return image_and_tags
