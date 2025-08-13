# The Deep Gym Project

This project was built for evaluating several deep learning models for the task of multilabel image classification.

## Installation
This project was built using Python 3.10.4. To install the required packages, run the following command from the project root directory:
```
pip install -r requirements.txt
```
 
## Running the project

### Prerequisites
First of all, make sure you have the train data in the correct locations:
* The annotation files should be in the `project/data/annotations/` directory.
* The train images should be in the `project/data/images/` directory.
* The unlabelled evaluation images, used for the final predictions, should be in the `project/data/eval_images/` directory.


### Running the code
The code run in three different modes (as described in the report): train, evaluation and prediction mode.
The mode can be specified with the `-mode` argument. The default mode is `train`.

#### Running on Puhti
The code should be launched from Puhti only using the `queuer.sh` script. From this script, the user can specify the set of
parameters that will be used to run the code. It's possible to specify a set of multiple values for the same parametr, 
and the script will take care of queuing a job for all the combinations of the parameters. This is very useful for 
trying out many hyperparameters with ease, minimizing the need of manually queuing jobs.

To ease the process of running the code in Puhti, the `queuer.sh` script makes use of another `run_batch.sh` script. 
The latter just forwards all the command line arguments given to it to the `main.py` script, and also provides some 
extra information for the Puhti system to process the job. But you should ignore it, don't ever modify it.

#### Running locally
If you want to run the code locally, you can use the `main.py` script directly. The script takes the following arguments:

```{bash}
-m INTEGER RANGE     Model version  [0<=x<=6]
-loss INTEGER RANGE  Loss function type [0<=x<=3]
-opt INTEGER RANGE   Optimizer type  [0<=x<=4]
-rgb BOOLEAN         If True, load image in RGB. Else, in Grayscale
-noe BOOLEAN         If True, exclude images with empty labels from training
-aug BOOLEAN         If true, enable data augmentation for training images
-b INTEGER           Batch size
-lr FLOAT            Learning rate
-e INTEGER           Number of epochs
-t FLOAT             Threshold for prediction
-s FLOAT RANGE       Fraction of dataset to load (for faster debug). [0<x<=1]
-split FLOAT RANGE   Fraction of data to use for training. [0<x<1]
-cache BOOLEAN       If true, cache images in memory for faster training
-test BOOLEAN        If False, use the the dev dataset also for testing.
-mode TEXT           Mode: [train, eval, pred]
-load PATH           Load model for evaluation or prediction. It needs to point to a folder containing a model.pt file
-id TEXT             If given, adds an id to the final folder name
-v BOOLEAN           Enable verbose output
--help               Show this message and exit.
```

If no arguments are given, the script will run with the default values specified in the `main.py` file.
We don't recommend using the default values, as they are set for quick testing purposes only, to speed up
development.


### Training a model
We have made 7 different models, check the file `models.py` for implementation details for each of them. The model 
selection is done with the `-m` argument, which takes an integer value in the range [0, 6]. The default value is 4,
which is the Dummy model that does not learn and always predicts no labels.

You should check the `getters.py` file to see the available models, loss functions and optimizers. For example, 
you can run the following command to start training using the Model5 with the BCE Loss and the SGD optimizer, which loads
the images in Grayscale format, do random augmentations on the test set and uses a 80/10/10 split for 
training/validation/testing:
```
python main.py -mode train -m 5 -loss 0 -opt 2 -split 0.8 -rgb 0 -aug 1
```

The program will output the training progress in the standard output. At the end of the training, the program will
save an extensive report under the `project/results/` directory, in a new folder named with a summary of the parameters
used for training. The report will contain plots for the training and validation accuracy and loss, confusion matrices,
per-label and aggregated statistical scores (precision, recall, f1-score), a prediction file for the test set and 
a checkpoint for the model's weights, that will be useful for further evaluation. Also, it saves the dictionary of 
command arguments passed as input, in a file named `args.json` inside the results folder.

**NOTE:** To greatly speed up the training process, we recommend adding the flag `-cache 1` to the command line. This
will cache the images in memory as they are loaded, so that they won't need to be loaded again at each epoch. Of course,
you should make sure to have enough ram available for this.

### Evaluating a model
This mode is useful for evaluating pre-trained models against a labelled dataset, and generating a report of its
performance. We used this mode mostly for development purposes, to recompute the results of already trained models 
after bugfixes or changes in the code. Here is an example:
```
python main.py -mode eval -load project/results/some_model_folder/
```
The output will be similar to the one generated by the training mode, but it will not contain the plots of the
training history.


### Predicting labels of unlabeled data
We used this mode to generate the final predictions for the evaluation images. The evaluation images need to be
inside the `project/data/eval_images/` directory. 

```
python main.py -mode pred -load project/results/some_model_folder/
```

The output will be a TSV file named `predictions.tsv` inside the results folder. The file specification
is described in the project instructions.