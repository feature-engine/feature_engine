# Notebooks with Demos of Feature-engine's Functionality

## How to run the notebooks

If you have jupyter and the Python numerical libraries installed, then you just need to do `pip install feature_engine`.

If you want to create a separate environment:

1) create an environment with the libraries indicated in example_requirements.txt

If using venv:
`python -m venv path/to/environment/feateng`

2) activate the environment:
`path/to/environment/feateng/Scripts\activate`

Directions 1 and 2 assume you are on windows. If using mac, directions are slightly different. Please check [venv website](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

3) install the required libraries:

`pip install requirements.txt`

4) add the new environment to the ipykernel:

`python -m ipykernel install --user --name feateng --display-name "feat_engine"`


