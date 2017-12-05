# online-meta-learning

This repository contains the code for "Online Learning of a Memory of
Learning Rates" ([Arxiv](https://arxiv.org/abs/1709.06709)).

## Setup

Please execute the following commands to setup a virtual environment with the correct dependencies.
```bash
# This script will install all dependencies, initially please answer y for upgrade dependencies.
# Answer y to install pytorch for your current platform.
# Answer y to install tensorflow for your current platform.
bash scripts/install_dependencies.sh
```

## Examples
Currently, we have example notebook for the pytorch and tensorflow backend available, more notebooks with mnist examples will be available soon.

```bash
bash scripts/start_jupyter.sh

# By default a browser window should open with the url localhost:8888.
# For pytorch please click on notebooks/example_rosenbrock_pytorch.ipynb
# For tensorflow please click on notebooks/example_rosenbrock_tf.ipynb
# Now you can run through the example notebook.
``` 
