Performance measurements
========================

Run the benchmark from the repository root:

.. code-block:: bash

   python backend/benchmarks/performance.py --sizes 1000 5000 10000 --epochs 50

Each size uses generated data, a 60/20/20 split and a two-layer MLP with 64
nodes per layer. Model construction precedes timing. Fitting includes the
validation pass, and prediction timing covers one forward pass on the test
split. The JSON output records configuration, software and machine architecture.

Memory scope
------------

The benchmark uses Python's tracemalloc during fitting. Its peak describes
traced Python allocations. TensorFlow native allocations, accelerator memory
and total process memory are outside this measurement.

Interpretation
--------------

A single timing per size is illustrative and does not establish asymptotic
complexity or a hardware-independent speed claim. TensorFlow tracing, cache
warm-up and machine load affect timing. Synthetic test metrics describe each
generated dataset and configuration; they are not accuracy estimates for
physical observations.

The command writes ``outputs/benchmark/summary.json``. The numerical
example's held-out metrics and learning curves can be reproduced separately:

.. code-block:: bash

   python -m fivedreg.demo --output-dir outputs/example
