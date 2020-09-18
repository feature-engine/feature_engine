.. -*- mode: rst -*-

Getting Started with Feature-engine on GitHub
=============================================

Feature-engine is hosted on `GitHub <https://github.com/solegalli/feature_engine>`_.

A typical contributing workflow goes like this:

1. **Find** a bug while using Feature-engine, **suggest** new functionality, or **pick up** an issue from our  `repo <https://github.com/solegalli/feature_engine/issues/>`_.
2. **Discuss** with us how you would like to get involved and your approach to resolve the issue.
3. Then, **fork** the repository into your GitHub account.
4. **Clone** your fork into your local computer.
5. **Code** the feature, the tests and update or add the documentation.
6. **Review** the code with one of us, who will guide you to a final submission.
7. **Merge** your contribution into the Feature-engine source code base.

It is important that we communicate right from the beginning, so we have a clear understanding of how you would like to get involved and what is needed to complete the task.

Forking the Repository
----------------------

When you fork the repository, you create a copy of Feature-engine's source code into your account, which you can edit. To fork Feature-engine's repository, click the **fork** button in the upper right corner of Feature-engine's GitHub page.

Once forked, follow these steps to set up your development environment:

1. Clone your fork into your local machine::

        $ git clone https://github.com/<YOURUSERNAME>/feature_engine

2. Create a virtual environment with the virtual environment tool of your choice.

3. Change directory into the cloned repository::

        $ cd feature_engine

4. Install Feature_engine in developer mode::

        $ pip install -e .

This will add Feature-engine to your PYTHONPATH so your code edits are automatically picked up, and there is no need to re-install the package after each code change.
    
5. Install the additional dependencies for tests and documentation::

        $ pip install -r test_requirements.txt
        $ pip install -r docs/requirements.txt

6. Make sure that your local master is up to date with the remote master.

7. Create a new branch where you will develop your feature::

    $ git checkout -b myfeaturebranch

Please give the branch a name that identifies which feature you are going to build.

8. If you haven't done so, set up an ``upstream`` remote from where you can pull the latest code changes occurring in the main Feature-engine repository::

    $ git remote add upstream https://github.com/solegalli/feature_engine.git
    $ git remote -v
    origin    https://github.com/YOUR_USERNAME/feature_engine.git (fetch)
    origin    https://github.com/YOUR_USERNAMEfeature_engine.git (push)
    upstream  https://github.com/solegalli/feature_engine.git (fetch)
    upstream  https://github.com/solegalli/feature_engine.git (push)

Now you are ready to start developing your feature.

Developing a new feature
------------------------

First thing, make a pull request (PR). The PR should be made from your feature_branch (in your fork), to Feature-engine's master branch in the main repository.

At this point, we should have discussed the contribution. But if we haven't, we will get in touch to do so.

Once your contribution contains the new code, the tests, and ideally the documentation, the review process will start. Likely, there will be some back and forth, until the final submission.

Once the submission is reviewed and provided the continuous integration tests have passed and the code is up to date with Feature-engine's master branch, we will be ready to "Squash and Merge" your contribution, into the ``master`` branch of Feature-engine. "Squash and Merge" combines all of your commits into a single commit which helps keep the history of the repository clean and tidy.

After a few features have been added to the master branch, by yourself and other contributors, we will merge master into a specific version branch, e.g. 0.6.X, to release in a new Feature-engine version to PyPI. Once your contribution has been merged into master, you will be listed as a Feature-engine contributor :)


Testing the code in the PR
--------------------------

You can test the code functionality either in your development environment or using tox. If you want to use tox:

1. Install tox in your development environment::

    $ pip install tox

2. Make sure you are in the repository folder, alternatively::

    $ cd feature_engine

3. Run the tests in tox::

    $ tox

If the tests pass, the local setup is complete.

If you want to know more about tox follow this `link <https://tox.readthedocs.io>`_. If you want to know why we prefer tox, this `article <https://christophergs.com/python/2020/04/12/python-tox-why-use-it-and-tutorial/>`_
will tell you everything ;)

If you prefer not to use tox, there are a few options. If you are using Pycharm:

1. In your project directory (where you have all the files and scripts), click with the mouse right button
on the folder "tests".

2. Select "Run pytest in tests".

3. Done!!

Sweet, isn't it?

You can also run the tests from your command line:

1. Open a command line and change into the repo directory.
2. Run::

    $ pytest

These command will run all the test scripts within the test folder. Alternatively, you can run specific scripts as follows:

1. Change into the tests folder::

    $ cd tests

2. Run a specific script, for example::

    $ pytest test_categorical_encoder.py

If running pytest without tox, that is in your development environment, make sure you have the test dependencies installed.
If not, from the root directory of the repo and in your development environment run::

    $ pip install -r test_requirements.txt

If tests pass, your code is functional. If not, try and fix the issue following the error messages. If stuck, get in touch.

Merging Pull Requests
---------------------

Only Core contributors have write access to the repository and can merge pull requests. If you are a core
contributor, some preferences for commit messages when merging in pull requests:

- Make sure to use the “Squash and Merge” option in order to create a Git history that is understandable.
- Keep the title of the commit short and descriptive; be sure it includes the PR # and the issue #.


After your PR is merged
-----------------------

Update your local fork::

    $ git checkout master
    $ git pull upstream master
    $ git push origin master

Finally, delete the old feature branch, both locally and on GitHub. Well done and thank you very much!

