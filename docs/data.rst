Data
====

Dataset format
--------------

The application accepts a NumPy ``.npz`` archive containing two arrays:

* ``X``: finite numerical features with shape ``(N, 5)``.
* ``y``: finite numerical targets with shape ``(N,)``.

Rows in the two arrays must correspond to the same observations. Object arrays,
missing values and infinities are not supported. Values must be within the
float32 range used by the model. Pickle files are not accepted.

Save arrays already available in a Python session with:

.. code-block:: python

   import numpy as np

   np.savez("dataset.npz", X=X, y=y)

Synthetic example
-----------------

The example generator draws five independent standard-normal features and forms
a nonlinear target:

.. math::

   x_j &\sim \mathcal{N}(0, 1), \qquad j=1,\ldots,5, \\
   y &= x_1^2 + 2x_2 + \sin(x_3) + x_4x_5 + \epsilon, \\
   \epsilon &\sim \mathcal{N}(0, \sigma^2).

The default dataset has 1,000 observations, random seed 42 and noise standard
deviation ``sigma = 0.1``. All variables are dimensionless. This generated
regression problem provides a reproducible example for training, prediction and
performance measurements.

Splitting and standardisation
------------------------------

The default split assigns 60% of observations to training, 20% to validation and
the remainder to testing, after shuffling with seed 42. For the default example,
the split sizes are 600, 200 and 200.

``StandardScaler`` is fitted on the training features only. Its means and scales
are then applied to validation, test and prediction inputs. Targets retain their
input scale. Prediction requests therefore use raw feature values in the same
units and column order as the uploaded ``X``; do not standardise them beforehand.

External data
-------------

Other regression datasets can be supplied as numerical arrays in the same NPZ
format. The course-provided dataset is not included: its accompanying materials
do not specify a redistribution licence. The repository includes no course data
or assignment files.
