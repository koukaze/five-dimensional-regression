Testing
=======

Backend checks
--------------

.. code-block:: bash

   python -m pytest backend/tests

Tests check data shapes and finite values, safe NPZ loading, reproducible
splits, train-only scaling, model parameter validation and the complete
upload–train–predict workflow. Prediction checks compare raw API inputs with
the same inputs transformed by the training scaler. Session checks cover
dataset replacement and model invalidation.

Frontend checks
---------------

.. code-block:: bash

   cd frontend
   npm run lint
   npm test
   npm run build

With both services running, load the example data, train a model and submit
a prediction. Also check the missing-model and invalid-input messages.

Documentation checks
--------------------

.. code-block:: bash

   python -m sphinx -W --keep-going -b html docs docs/_build/html

The GitHub Actions workflow defines backend tests, frontend lint/build and
documentation checks for pushes and pull requests.
