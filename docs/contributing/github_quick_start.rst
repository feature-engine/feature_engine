.. -*- mode: rst -*-

Getting Started with Feature-engine on GitHub
=============================================

Feature-engine is hosted on `GitHub <https://github.com/solegalli/feature_engine>`_.

A typical contributing workflow goes like this:

1. **Find** a bug while using Feature-engine, **suggest** new functionality, or **pick up** an issue from our  `repo <https://github.com/solegalli/feature_engine/issues/>`_.
2. **Discuss** with us how you would like to get involved and your approach to resolve the issue.
3. Then, **fork** the repository into your GitHub account.
4. **Clone** your fork into your local computer.
5. **Code** the feature, the tests and ideally as well, the documentation.
6. **Review** the code with one of us, who will guide you to a final submission.
7. **Merge** your contribution into the Feature-engine source code base.

It is important that we communicate right from the beginning, so we have a clear understanding of how you would like to get involved and what is needed to complete the task.

Forking the Repository
----------------------

When you fork the repository, you create a copy of Feature-engine's source code into your account, which you can edit. To fork Feature-engine's repository, click the **fork** button in the upper right corner of Feature-engine's GitHub page.

Once forked, follow these steps to set up your development environment:

1. Clone your fork into your local machine.::

        $ git clone https://github.com/<YOURUSERNAME>/feature_engine

2. Create a virtual environment with the virtual environment tool of your choice.

3. Change directory into the cloned repository.::

        $ cd feature_engine

4. Install Feature_engine in developer mode.::

        $ pip install -e .

This will add Feature-engine to your PYTHONPATH so your code edits are automatically picked up, and there is no need to re-install the package after each code change.
    
5. Install the additional dependencies for tests and documentation.::

        $ pip install -r test_requirements.txt
        $ pip install -r docs/requirements.txt

6. Checkout and switch to the develop branch.

Feature-engine's repository has a ``develop`` branch where we develop the new functionality before the next version is released. Switch to the develop branch as follows.::

        $ git fetch
        $ git checkout develop

7. Create a new branch where you will develop your feature.::

    $ git checkout -b myfeaturebranch develop

Please give the branch a name that identifies which feature you are going to build. 

8. If you haven't done so, set up up an ``upstream`` remote from where you can pull the latest code changes occurying in the main Feature-engine repository::

    $ git remote add upstream https://github.com/solegalli/feature_engine.git
    $ git remote -v
    origin    https://github.com/YOUR_USERNAME/feature_engine.git (fetch)
    origin    https://github.com/YOUR_USERNAMEfeature_engine.git (push)
    upstream  https://github.com/solegalli/feature_engine.git (fetch)
    upstream  https://github.com/solegalli/feature_engine.git (push)

Now you are ready to start developing your feature.

Developing a new feature
------------------------

9. First thing, make a pull request (PR). The PR should be made from your feature_branch (in your fork), to Feature-engine's develop branch in the main repository.

At this point, we should have discussed the contribution. But if we haven't, we will get in touch to do so.

Once your contribution contains the new code, the tests, and ideally the documentation, the review process will start. Likely, there will be some back and forth, until the final submission.

One the submission is reviewd and provided the continuous integration tests have passed and the code is up to date with Feature-engine's develop branch, we will be ready to "Squash and Merge" your contribution, into the ``develop`` branch of Feature-engine. "Squash and Merge" combines all of your commits into a single commit which helps keep the history of the repository clean and tidy. 

After a few features have been developed in the develop branch, by yourself and others, it will be merged into master and released in a new Feature-engine version to PyPi. Once your contribution has been merged into master, you will be listed as a Feature-engine contributor :)

After your PR is merged
-----------------------

Update your local fork::

    $ git checkout develop
    $ git pull upstream develop
    $ git push origin develop

Finally, delete the old feature branch, both locally and on GitHub. Well done and thank you very much!

