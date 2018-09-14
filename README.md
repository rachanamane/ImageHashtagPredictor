#Instagram Hashtag Predictor
The raw dataset used is large in size. If you need to train the model from with the dataset, please email me. 

##Tech
[Tensorflow](https://www.tensorflow.org/): Install tensorflow [here] (https://www.tensorflow.org/install/) 

Need to run the following before executing any scripts:
```
export PYTHONPATH=/home/<path-to-root>/ImageHashtagPredictor/
```
Change the Shared -> Flags file suitably. All the paths and parameter inputs are in the flags file. 
Execute these commands:
```
git clone https://github.com/rachanamane/ImageHashtagPredictor.git
cd ImageHashtagPredictor
python preprocess/createHashtagFile.py
python preprocess/writeTFRecords.py
python model/runmodel.py (In case you are training the model and not using the pre-trained checkpoints)
python model/evaluatemodel.py
python model/predict.py --image_path=<insert_image_path_here>
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
