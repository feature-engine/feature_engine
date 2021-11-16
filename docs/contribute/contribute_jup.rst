.. -*- mode: rst -*-

Contribute Jupyter notebooks
============================

We created a collection of Jupyter notebooks that showcase the main functionality of
Feature-engine's transformers. We link these notebooks throughout the main documentation
to offer users more examples and details about transformers and how to use them.

**Note** that the Jupyter notebooks are hosted in a separate
`Github repository <https://github.com/feature-engine/feature-engine-examples>`_.

Here are some guidelines on how to add a new notebook or update an existing one. The
contribution workflow is the same we use for the main source code base.

Jupyter contribution workflow
-----------------------------

1. Fork the `Github repository <https://github.com/feature-engine/feature-engine-examples>`_.
2. Clone your fork into your local computer: `git clone https://github.com/<YOURUSERNAME>/feature-engine-examples.git`.
3. Navigate into the project directory: `cd feature-engine-examples`.
4. If you haven't done so yet, install feature-engine: `pip install feature_engine`.
5. Create a feature branch with a meaningful name: `git checkout -b mynotebookbranch`.
6. Develop your notebook
7. Add the changes to your copy of the fork: `git add .`, `git commit -m "a meaningful commit message"`, `git pull origin mynotebookbranch`.
8. Go to your fork on Github and make a PR to this repo
9. Done

The review process for notebooks is usually much faster than for the main source code base.

Jupyter creation guidelines
---------------------------

If you want to add a new Jupyter notebook, there are a few things to note:

- Make sure that the dataset you use is publicly available and with a clear license that it is free to use
- Do not upload datasets to the repository
- Add instructions on how to obtain and prepare the data for the demo
- Throughout the notebook, add guidelines on what you are going to do next, and what is the conclusion of the output

That's it! Fairly straightforward.

We look forward to your contribution :)