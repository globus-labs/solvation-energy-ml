# Predicting artial Charges

We used the DFT partial charges in the molecular repressentation passed to our MPNN.
The models worked well, but are not very usable because they require DFT to compute the partial charges.
Here, we explore predicting the partial charges with MPNNs so that we can use them as features for our model in place of DFT charges.
