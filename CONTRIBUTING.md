# Contributing

Install `python -m pip install -e '.[test,notebooks]'`, then run
`python -m pytest -q` and `python scripts/execute_notebooks.py`.

Keep algorithms in `bvp/` and experiments in notebooks or `examples/`.
Use English identifiers and docstrings. Explain boundary coefficient
conventions, mesh counts, initial guesses and stopping rules in each example.
Add a manufactured solution or an independent reference and check mesh
convergence. Test failures as well as successful solves. Never claim a speed
advantage from one timing or confuse a small residual with solution error.
Do not include private course documents, names, local file links or credentials.
