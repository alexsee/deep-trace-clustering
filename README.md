# DeepCluster: Learning Good Representations for Trace Clustering

## Setup
Use Miniconda for the easiest way to setup an environment.

### Using Miniconda
1. 1. Install [Miniconda](https://conda.io/miniconda.html) (make sure to use a Python 3 version)
2. After setting up miniconda you can make use of the `conda` command in your command line (Powershell, CMD, Bash)
3. We suggest that you set up a dedicated environment for this project by running `conda env create -f environment.yml`
    * This will setup a virtual conda environment with all necessary dependencies.
    * If your device does have a GPU replace `tensorflow` with `tensorflow-gpu` in the `environement.yml`
4. Depending on your operating system you can activate the virtual environment with `conda activate replearn` on Linux and macOS, and `activate ad` on Windows (`cmd` only).
5. If you want to make use of a GPU, you must install the CUDA Toolkit. To install the CUDA Toolkit on your computer refer to the [TensorFlow installation guide](https://www.tensorflow.org/install/install_windows).
6. If you want to quickly install the `replearn` package, run `pip install -e .` inside the root directory.
7. Now you can start the notebook server by `jupyter notebook notebooks`.

## Jupyter Notebooks
Check the `notebooks` directory for example Jupyter Notebooks.