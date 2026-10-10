Installation
============

Use Python 3.11–3.13 and Node.js 22 or 24. Commands start in the repository root.

Backend
-------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -m pip install --no-deps -e ./backend
   uvicorn fivedreg.main:app --host 127.0.0.1 --port 8000

Frontend
--------

In another terminal:

.. code-block:: bash

   cd frontend
   npm ci
   npm run dev

Open ``http://localhost:3000``. Browser requests use the same-origin ``/api``
proxy. Its server-side ``BACKEND_URL`` defaults to ``http://127.0.0.1:8000``. Set it before building to change the target.

For a production frontend build:

.. code-block:: bash

   npm run build
   npm start

The backend keeps one dataset and one model in memory. Run one Uvicorn worker
for this local application. Restarting the backend clears both.

Docker Compose
--------------

With Docker running:

.. code-block:: bash

   docker compose up --build

Both ports bind to the local host. The frontend resolves the backend through
the Compose service name; browsers continue to use ``/api``. Stop with
``docker compose down``.

Documentation
-------------

.. code-block:: bash

   python -m pip install -r requirements-dev.txt
   python -m sphinx -W --keep-going -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` for the local documentation.
