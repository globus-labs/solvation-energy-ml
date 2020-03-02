# Training MPNN Model

The notebooks in this folder are all associated with training and evaluating message-passing neural networks for predicting solvation energy.
The collection of notebooks includes those for making the networks and analyzing the results of the training.

The actual script for training the model is `train_models.py`, which requires Horovod to run propertly.
We recommend two routs for using these files:

1. Downloading the trained MPNN results. They will be made available on the MDF and will allow you to run all of the notebooks without 
2. Training the models on your ownw ithout Horovod. We will prepare a simple training script for using the model files created here 
   that avoids all of the cruft we used to organize model training.
