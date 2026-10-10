Using the application
=====================

Upload, train, predict
----------------------

1. Open **Upload** and use the included example data, or select a numeric NPZ
   file containing five features and a scalar target.
2. Open **Train** and choose the number of hidden layers, nodes per layer,
   learning rate, epochs and model seed.
3. Inspect training, validation and test MSE, MAE and R².
4. Open **Predict** and enter five values in the same feature order and units
   as the uploaded data. Preprocessing is applied by the backend.

A valid upload replaces the active dataset and clears its trained model.
Invalid uploads leave the current session intact. The application has one
shared local session; multiple browser tabs use that same state.

Training method
---------------

The dataset is shuffled with split seed 42 and divided into 60% training,
20% validation and 20% test observations. A StandardScaler is fitted only to
the training features, then applied to all splits and future prediction inputs.
Targets keep their supplied scale.

Each hidden layer uses a ReLU activation, followed by a linear scalar output.
Adam minimises mean squared error for the configured fixed number of epochs.
Validation loss is monitored; evaluation uses the final epoch's weights.
There is no validation-based checkpoint selection or early stopping.

The validation metrics support model comparison. Test metrics describe the
current trained model and should not be used to tune successive configurations.
The MLP is a regression model: predictions outside the feature range are
extrapolations with no accuracy guarantee.

Default configuration
---------------------

* Hidden layers: 2, with 64 nodes in each layer.
* Initial learning rate: 0.001.
* Epochs: 100, using mini-batches of 32 observations.
* Model seed: 42; split seed: 42.

MSE has squared target units, MAE has target units, and R² is dimensionless.
