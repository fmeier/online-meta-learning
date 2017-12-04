# online-meta-learning

This repository contains the code for "Online Learning of a Memory of
Learning Rates" ([Arxiv](https://arxiv.org/abs/1709.06709)).

## Setup

Please execute the following commands to setup a virtual environment with the correct dependencies.
```bash
# This script will install all dependencies, initially please answer y for upgrade dependencies.
# Answer y to install pytorch for your current platform.
bash scripts/install_dependencies.sh
```

## Examples
Currently, we have example notebook for the pytorch backend available, more notebooks also with a tensorbackend will be added soon.

```bash
# As a first step source the virtual environment.
source vpy/bin/activate
# Now you can start the jupyter notebook backend
jupyter notebook

# By default a browser window should open with the url localhost:8888.
# Please click on notebooks/example_rosenbrock_pytorch.ipynb
# Now you can run through the example notebook.
``` 
