API reference
=============

Interactive OpenAPI documentation is available at
``http://localhost:8000/docs``.

GET /health
-----------

Returns ``{"status": "ok"}`` without changing session state.

GET /status
-----------

Returns ``dataset_loaded``, ``model_trained``, ``n_samples``,
``split_sizes`` and ``training``.

POST /upload
------------

Multipart upload with one ``file`` field containing numeric NPZ arrays
``X`` and ``y``. See :doc:`data` for the schema. Valid data are split,
standardised and stored in memory. This replaces the dataset and clears the
trained model. The response includes a message, row count and split sizes.

POST /train
-----------

Example JSON body:

.. code-block:: json

   {"n_layers": 2, "n_nodes": 64, "lr": 0.001, "max_epochs": 100, "seed": 42}

Limits are 1–5 hidden layers, 1–256 nodes per layer, 1–500 epochs, a learning
rate in (0, 0.1], and a model seed in [0, 2147483647]. Training uses the active
dataset and a fixed split seed of 42. The response contains final-model
``metrics`` for train/validation/test, Keras ``history``, ``configuration``
and ``split_sizes``.

POST /predict
-------------

Example JSON body:

.. code-block:: json

   {"X": [0.1, 0.2, 0.3, 0.4, 0.5]}

The five raw values are transformed using the retained training scaler.
The response is ``{"prediction": <float>}``.

Errors and session state
------------------------

Missing data or a model returns HTTP 400. Invalid request parameters return
HTTP 422. Invalid datasets return HTTP 400; uploads over 20 MiB return HTTP 413.
An operation that overlaps another state-changing operation returns HTTP 409.
There is one in-memory workflow per backend process.

Python interfaces
-----------------

* ``fivedreg.data.load_dataset(path, train_ratio=0.6, val_ratio=0.2, seed=42)``
  returns a DatasetSplit with standardised features, targets, row indices and scaler.
* ``fivedreg.data.split_dataset(X, y, ...)`` prepares the same structure from arrays.
* ``MLP(...).fit(X, y, validation_data=(X_val, y_val))`` trains on standardised inputs.
* ``MLP.predict(X)`` returns a one-dimensional prediction array.
* ``MLP.metrics(X, y)`` returns MSE, MAE and R²; undefined R² is null.
* ``MLP.evaluate(X, y)`` returns the pair (MSE, MAE).

The MLP retains final-epoch weights and exposes the Keras training history.
