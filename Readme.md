# Predicting answer correctness of e-learners with a deep learning transformer model

This is my entry to the Kaggle challenge ["Riiid Answer Correctness
Prediction"](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/193250)

There are 4 distinct Kaggle kernels:

- `saint-plus/`: defines the transformer model,
- `saint-plus-preprocessing/`: preprocesses the data to put it into a hdf5
  format that will be fast to read when training the model,
- `saint-plus-train/`: trains the model, reading the data produced by the
  previous kernel
- `saint-plus-evaluate/`: evaluates the trained model 

Since these datasets use each other's outputs, they have to be properly imported
in the Kaggle interface (e.g. `saint-plus-train` must import
`saint-plus-preprocessing`). 

The kernels `saint-plus-train` and `saint-plus-evaluate` must import
`saint-plus` as a script.

# Resources

## Challenge

Challenge website
https://www.ednetchallenge.ai/

A notebook emulating the time-series API to test the submission procedure locally.
https://www.kaggle.com/its7171/time-series-api-iter-test-emulator

Starter notebook with instructions on how to use the API
https://www.kaggle.com/sohier/competition-api-detailed-introduction

Information about the test set in
https://www.kaggle.com/c/riiid-test-answer-prediction/data

    The API provides user interactions groups in the order in which they
    occurred. Each group will contain interactions from many different users,
    but no more than one task_container_id of questions from any single user.
    Each group has between 1 and 1000 users.

The test set has new users but no new questions.
https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/191106
https://www.kaggle.com/sohier/competition-api-detailed-introduction#1064314


## Dataset

Data set paper
https://arxiv.org/abs/1912.03072

Data set description and access on github
https://github.com/riiid/ednet

Tutorial on reading large datasets:
https://www.kaggle.com/rohanrao/tutorial-on-reading-large-datasets      

On the difference between task_container_id and bundle_id.
https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/190828


## Saint and Saint+

Saint+ paper
https://arxiv.org/abs/2010.12042

Saint+ discussion on kaggle
https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/193250

Saint benchmark discussion on kaggle (interesting tips on implementation)
https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/195632


## Transformers

Transformer paper: Attention is all you need
https://arxiv.org/abs/1706.03762

Links to explanations on transformers
https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/192138

The illustrated transformer
https://jalammar.github.io/illustrated-transformer/

Tensorflow Tutorial on transformer
https://www.tensorflow.org/tutorials/text/transformer

ADAM optimizer
https://arxiv.org/pdf/1412.6980.pdf

Transformer example with Keras
https://www.tensorflow.org/tutorials/text/transformer

Training tips for the Transformer model
https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf


## Parallelisation

Exemple d'utilisation de GPU ou TPU
https://www.kaggle.com/mgornergoogle/five-flowers-with-keras-and-xception-on-tpu#Kaggle-dataset-access
