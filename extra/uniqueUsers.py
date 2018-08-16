import argparse
from os import listdir
from os.path import isfile, join, isdir

def process_directory(dir, imagesByUser):
    files = listdir(dir)
    for file in files:
        filePath = join(dir, file)
        if isfile(filePath):
            if file.endswith(".jpg"):
                userId = file.split("_")[0]
                if userId in imagesByUser:
                    imagesByUser[userId] += 1
                else:
                    imagesByUser[userId] = 1
        elif isdir(filePath) and file != "extra":
            process_directory(filePath, imagesByUser)

def main():
    imagesByUser = {}
    process_directory(FLAGS.dataset_dir, imagesByUser)
    print("Total Users: %s" % len(imagesByUser))
    freqMap = {}
    for key in imagesByUser:
        val = imagesByUser[key]
        if val in freqMap:
            freqMap[val] += 1
        else:
            freqMap[val] = 1
    print("\nImages NumUsers")
    for key, value in sorted(freqMap.items()):
        print("%6s %-5s" % (key, value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--dataset_dir',
      type=str,
      default='/Users/namitr/tfprograms/dataset',
      help='Absolute path to dataset.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()