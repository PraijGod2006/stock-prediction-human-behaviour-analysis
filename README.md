# Python and Jupyter Project Notes

Project location: `D:\CODE\rajasthani`

## Current setup

- Notebook: `pytorch.ipynb`
- Virtual environment: `D:\CODE\rajasthani\shitimon`
- Environment Python: `D:\CODE\rajasthani\shitimon\Scripts\python.exe`
- Registered kernel name: `Python (.venv)`

The Jupyter kernel registration file is stored in:

```text
C:\Users\Praji\AppData\Roaming\jupyter\kernels\venv
```

This is normal. The registration file is very small. It points to the real Python executable in the `D:` drive, where the environment, packages, and their files are stored.

## Activate the environment

Open a terminal in this project folder and run:

### Command Prompt

```cmd
cd /d D:\CODE\rajasthani
shitimon\Scripts\activate
```

### PowerShell

```powershell
Set-Location D:\CODE\rajasthani
.\shitimon\Scripts\Activate.ps1
```

When activated, the terminal should show `(shitimon)` at the beginning of the prompt.

## Install packages

Always install packages through the active environment:

```cmd
python -m pip install --upgrade pip
python -m pip install ipykernel pandas numpy matplotlib scikit-learn
```

If the notebook kernel disappears or points to the wrong Python, register it again:

```cmd
python -m pip install ipykernel
python -m ipykernel install --user --name=rajasthani --display-name "Python (rajasthani)"
```

Then open the notebook, choose **Select Kernel**, and select **Python (rajasthani)**.

## Verify the notebook kernel

Run this in a notebook cell:

```python
import sys
print(sys.executable)
```

It should print a path containing:

```text
D:\CODE\rajasthani\shitimon\Scripts\python.exe
```

You can also check installed packages from a notebook cell:

```python
%pip list
```

## Store large datasets on the D drive

Create separate folders for data and outputs:

```cmd
mkdir data
mkdir outputs
mkdir models
```

Recommended layout:

```text
rajasthani/
|-- pytorch.ipynb
|-- README.md
|-- shitimon/       Virtual environment
|-- data/            Downloaded datasets
|-- outputs/         Charts and predictions
|-- models/          Saved model files
```

Download large datasets into `D:\CODE\rajasthani\data` or another folder on the `D:` drive. This will not use significant space on `C:`. Check that `D:` has enough free space before downloading.

Use relative paths in notebooks so projects remain portable:

```python
from pathlib import Path

DATA_DIR = Path("data")
file_path = DATA_DIR / "dataset.csv"
```

Avoid putting datasets inside the virtual environment. The environment should contain Python packages only.

## Important storage note

- Python itself may be installed on `C:`. That is usually fine.
- The virtual environment and packages are on `D:` in this project.
- Jupyter kernel registration is on `C:` and is only a small configuration file.
- Downloaded datasets, model checkpoints, and notebook outputs should be stored on `D:`.
- Do not move or rename `shitimon` after registering the kernel unless you recreate the environment or register the kernel again.

## Before starting a new project

1. Create the project folder on the drive with enough space.
2. Create a virtual environment inside that project.
3. Activate it before installing packages.
4. Install `ipykernel` and register a clearly named kernel.
5. Keep datasets, models, and outputs outside the virtual environment.
6. Add large files to `.gitignore` if the project uses Git.

## Useful commands

```cmd
python --version
where python
python -m pip list
jupyter kernelspec list
```

`where python` should show the project environment first when it is activated.
