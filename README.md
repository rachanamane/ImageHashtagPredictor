# Instagram Hashtag Predictor
The raw dataset used is large in size. If you need to train the model from with the dataset, please to rachanamane93@gmail.com 

## Tech
[Tensorflow](https://www.tensorflow.org/): Install tensorflow [here] (https://www.tensorflow.org/install/) 

Need to run the following before executing any scripts:
```
export PYTHONPATH=/home/<path-to-root>/ImageHashtagPredictor/
```

Download the following folder which contains:
A. generated
    1. Checkpoints
        The checkpoints correspond to model trained on 42000 images run with 3 epochs.
    2. TF Records
        Tensorflow records for evaluation dataset (7000 images)
    3. Compiled list of all the hashtag labels
    4. User hitory output file
B. prediction
    Prediction dataset

```
https://drive.google.com/drive/folders/1rDS3h-42-6I1LIFfkkfrWa0CSQj5djS_?usp=sharing
```

Modify file Shared -> Flags 
1. tfrecords_dir : '/home/<path-to-root>/tfprograms/generated/tfrecords'
2. train_checkpoint_dir : '/home/<path-to-root>/tfprograms/generated/checkpoints'
3. user_history_output_file : '/home/<path-to-root>/tfprograms/generated/user_history.txt'
3. model_eval_shards : If you want to evaluate the full eval dataset, set the number of shards to 14. 
   One shard has 500 images.
```
Change the Shared -> Flags file suitably. All the paths and parameter inputs are in the flags file. 
Execute these commands:
```
git clone https://github.com/rachanamane/ImageHashtagPredictor.git
cd ImageHashtagPredictor
python preprocess/createHashtagFile.py
python preprocess/writeTFRecords.py
python model/runmodel.py 
python model/evaluatemodel.py
python model/predict.py --image_path=<insert_image_path_here>
```

Following are the commands to train the model

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
