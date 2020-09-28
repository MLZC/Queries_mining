# To-do

- [x] data standardization

---
# Environment preparetion

1. Create virtual environment
    > conda create -n queries_mining
<!-- 2. install conda dependencies
    > conda install --yes --file requirements_conda.txt -->
2. For **PowerShell** user:
    - open powershell as **administrator** (very importand)
        ```shell
        conda init powershell
        ```
    Repon powershell and activate our environment.
    > conda activate queries_mining
    
    **For cmd user: Go to step 3 directly.**
3. install pip dependencies
    > pip install -r requirements
---

# Run project

1. Add a new system variable:

    > MKL_THREADING_LAYER=GUN

    environment variables-> system variables-> new

    ```
    name=MKL_THREADING_LAYER
    value=GUN
    ```
    and restart all terminal(powershell and cmd) 
2. Go to the root director of the project
    > python run.py

**Note:**

If you have any error in step 2, you should openpowershell as **administrator** and run the following command:
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned

And choose **y**, **n** is default.
