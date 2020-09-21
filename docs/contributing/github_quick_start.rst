.. -*- mode: rst -*-

Getting Started with Feature-engine on GitHub
=============================================

Feature-engine is hosted on `GitHub <https://github.com/solegalli/feature_engine>`_.

A typical contributing workflow goes like this:

1. **Find** a bug while using Feature-engine, **suggest** new functionality, or **pick up** an issue from our `repo <https://github.com/solegalli/feature_engine/issues/>`_.
2. **Discuss** with us how you would like to get involved and your approach to resolve the issue.
3. Then, **fork** the repository into your GitHub account.
4. **Clone** your fork into your local computer.
5. **Code** the feature, the tests and update or add the documentation.
6. Make a **Pull Request (PR)** with your changes.
7. **Review** the code with one of us, who will guide you to a final submission.
8. **Merge** your contribution into the Feature-engine source code base.

It is important that we communicate right from the beginning, so we have a clear understanding of how you would like to get involved and what is needed to complete the task.

Forking the Repository
----------------------

When you fork the repository, you create a copy of Feature-engine's source code into your account, which you can edit. To fork Feature-engine's repository, click the **fork** button in the upper right corner of Feature-engine's GitHub page.


Setting up the Development Environment
--------------------------------------

Once you forked the repository, follow these steps to set up your development environment:

1. Clone your fork into your local machine::

    $ git clone https://github.com/<YOURUSERNAME>/feature_engine

2. Set up an ``upstream`` remote from where you can pull the latest code changes occurring in the main Feature-engine repository::

    $ git remote add upstream https://github.com/solegalli/feature_engine.git
    $ git remote -v
    origin    https://github.com/YOUR_USERNAME/feature_engine.git (fetch)
    origin    https://github.com/YOUR_USERNAMEfeature_engine.git (push)
    upstream  https://github.com/solegalli/feature_engine.git (fetch)
    upstream  https://github.com/solegalli/feature_engine.git (push)

Keep in mind that Feature-engine is being actively developed, so you may need to update your fork regularly. See below for tips on **Keeping your fork up to date**.

3. Optional but highly advisable: Create a virtual environment. Use any virtual environment tool of your choice. Some examples include:

    1. `venv <https://docs.python.org/3/library/venv.html>`_
    2. `conda environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_

4. Change directory into the cloned repository::

        $ cd feature_engine

5. Install Feature_engine in developer mode::

        $ pip install -e .

This will add Feature-engine to your PYTHONPATH so your code edits are automatically picked up, and there is no need to re-install the package after each code change.
    
6. Install the additional dependencies for tests and documentation::

        $ pip install -r test_requirements.txt
        $ pip install -r docs/requirements.txt

7. Make sure that your local master is up to date with the remote master::

        $ git pull --rebase upstream master

If you just cloned your fork, your local master should be up to date. If you cloned your fork a time ago, probably the main repository had some code changes. To sync your fork master to the main repository master, read below the section **Keeping your fork up to date**.

8. Create a new branch where you will develop your feature::

    $ git checkout -b myfeaturebranch

There are 3 things to keep in mind when creating a feature branch. First, give the branch a name that identifies the feature you are going to build. Second, make sure you checked out your branch from master branch. Third, make sure your local master was updated with the upstream master.

9. Once your code is ready, commit your changes and push your branch to your fork::

    $ git add .
    $ git commit -m "my commit message"
    $ git push origin myfeaturebranch

This will add a new branch to your fork. In the commit message, be succint, describe what is being added and if it resolves an issue, make sure to **reference the issue in the commit message** (you can also do this from Github).

10. Go to your fork in Github, you will see the branch you just pushed and next to it a button to create a PR. Go ahead and create a PR from your feature branch to Feature_engine's master branch.


Developing a New Feature
------------------------

First thing, make a pull request (PR). Once you have written a bit of code for your new feature, or bug fix, or example, or whatever task you are working on, make a PR. The PR should be made from your feature_branch (in your fork), to Feature-engine's master branch in the main repository.

When you develop a new feature, or bug, or any contribution, there are a few things to consider:
    
    1. Make regular code commits to your branch, locally.
    2. Give clear messages to your commits, indicating which changes were made at each commit (use present tense)
    3. Try and push regularly to your fork, so that you don't lose your changes, should a major catastrophe arise
    4. If your feature takes some time to develop, make sure you rebase upstream/master onto your feature branch


Once your contribution contains the new code, the tests, and ideally the documentation, the review process will start. Likely, there will be some back and forth until the final submission.

Once the submission is reviewed and provided the continuous integration tests have passed and the code is up to date with Feature-engine's master branch, we will be ready to "Squash and Merge" your contribution into the ``master`` branch of Feature-engine. "Squash and Merge" combines all of your commits into a single commit which helps keep the history of the repository clean and tidy.

Once your contribution has been merged into master, you will be listed as a Feature-engine contributor :)


Testing the Code in the PR
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


Keeping your Fork up to Date
----------------------------

When you're collaborating using forks, it's important to update your fork to capture changes that have been made by other collaborators.

If your feature takes a few days or weeks to develop, it may happen that new code changes are made to Feature_engine's master branch by other contributors. Some of the files that are changed maybe the same files you are working on. Thus, it is really important that you pull and rebase the upstream master into your feature branch, fairly often. To keep your branches up to date:

1. Check out your local master::

    $ git checkout master

If your feature branch has uncommited changes, it will ask you to commit or stage those first.

2. Pull and rebase the upstream master on your local master::

    $ git pull --rebase upstream master

Your master should be a copy of the upstream master. If was is not, there may appear some conflicting files. You will need to resolve these conflicts and continue the rebase.

3. Pull the changes to your fork::

    $ git push -f origin master

The previous command will update your fork so that your fork's master is in sync with Feature-engine's master. Now, you need to rebase master onto your feature branch.

4. Check out your feature branch::

    $ git checkout myfeaturebranch

5. Rebase master onto it::

    $ git rebase master

Again, if conflicts arise, try and resolve them and continue the rebase. Now you are good to go to continue developing your feature.


Merging Pull Requests
---------------------

Only Core contributors have write access to the repository, can review and can merge pull requests. Some preferences for commit messages when merging in pull requests:

- Make sure to use the “Squash and Merge” option in order to create a Git history that is understandable.
- Keep the title of the commit short and descriptive; be sure it includes the PR # and the issue #.


After your PR is merged
-----------------------

Update your local fork (see section **Keeping your fork updated**) and delete the feature branch.

Well done and thank you very much for your support!


Releases
--------

After a few features have been added to the master branch by yourself and other contributors, we will merge master into a release branch, e.g. 0.6.X, to release a new version of Feature-engine to PyPI. 