# To-do

- [x] data standardization

---
# Environment preparetion

1. Create virtual environment
    > conda create -n queries_mining python=3.6
<!-- 2. install conda dependencies
    > conda install --yes --file requirements_conda.txt -->
2. For **PowerShell** user:
    - open powershell as **administrator** (very importand)
        ```shell
        conda init powershell
        ```
    - Repon powershell and activate our environment.
        > conda activate queries_mining
    
    **For cmd user: conda activate queries_mining directly.**

    > conda activate queries_mining
3. Add channel conda-forge
    ```
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    ```
4. Install requirements
    > conda install --yes --file requirements.txt

    or

    > pip install -r requirements.txt 
---

# Run project

Go to the root director of the project

Run on the mini set
> python run.py 1 > 2k.txt

Run on the whole set:
> python run.py 0 > 10w.txt

# keras and tensorflow version

If keras==2.4.3 is not compatible with tensorflow==1.14.0, then try keras==2.0.8.

> pip install keras==2.0.8

**Note:**

If you have any error in step 2, you should openpowershell as **administrator** and run the following command:
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned

And choose **y**, **n** is default.
