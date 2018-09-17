# Instagram Hashtag Predictor
Nowadays, users upload and share images on social networks with hashtags (especially on
Instagram). There are a many really good machine learning models available (e.g. inception
model published by Google) for image recognition based on the pixels in the image. Hashtags
used by real users on Instagram are much more varied and nuanced at describing the image
being shared on Instagram. For example, an image of spaghetti could be shared with the
hashtags #italiandinner or #yummy. There are various factors apart from the pixels in the
image that can be leveraged to predict the hashtags that an user can use when uploading
an image. This project aims at leveraging that information, more specifically leveraging the
history of hashtags used by the user and globally trending hashtags in addition to the pixels
in the image to train a neural network for hashtag prediction.

## Prerequisites

1. Python v2.7: Install [here](https://www.python.org/downloads/release/python-2715/)

2. [Tensorflow](https://www.tensorflow.org/): Install tensorflow [here](https://www.tensorflow.org/install/) 

3. Need to run the following before executing any scripts:
```
export PYTHONPATH=/home/<path-to-root>/ImageHashtagPredictor/
```

## Pre-trained model with sample prediction images
Download the following folder which contains:
1. generated
    - Checkpoints
        The checkpoints correspond to model trained on 42000 images run with 3 epochs.
    - TF Records
        Tensorflow records for evaluation dataset (7000 images)
    - Compiled list of all the hashtag labels
    - User hitory output file
2. prediction
    Prediction dataset

```
https://drive.google.com/drive/folders/16xRHMzoStlXET9Sptm4zKQpjI-EV0fP2?usp=sharing
```

### Modify file Shared -> Flags 
1. tfrecords_dir : '/home/<path-to-root>/tfprograms/generated/tfrecords'
2. train_checkpoint_dir : '/home/<path-to-root>/tfprograms/generated/checkpoints'
3. user_history_output_file : '/home/<path-to-root>/tfprograms/generated/user_history.txt'
3. model_eval_shards : If you want to evaluate the full eval dataset, set the number of shards to 14. 
   One shard has 500 images.
   
### Evaluate pre-trained model 

```
git clone https://github.com/rachanamane/ImageHashtagPredictor.git
cd ImageHashtagPredictor
python model/evaluatemodel.py
python model/predict.py --image_path=<insert_image_path_here>
```

## Author
* Rachana Mane   

## Labels set for the model
```
The hashtag labels are following
     0 petstagram
     1 cute
     2 foodie
     3 dogsofinstagram
     4 sweet
     5 goodfood
     6 italiandinner
     7 breakfast
     8 chicken
     9 dogoftheday
    10 yummy
    11 instapets
    12 icecream
    13 cats
    14 mexicanfood
    15 animal
    16 adorable
    17 lovecats
    18 catstagram
    19 sandwich
    20 dessert
    21 eggs
    22 kitty
    23 pasta
    24 meow
    25 ilovemydog
    26 delish
    27 dog
    28 foodlover
    29 cake
    30 pizza
    31 puppylove
    ```
