# FitForBlochGr-neisenTheory

Guide line for run the fit algorithm.

## 1. Install python version 3.8 and required libraries in requirements.txt

## 2. Put the data in the data foler
Note: The structure of the data need to follow the demo.csv file. 
For example:

Column 1: Temprature at Kevin scale
Column 2: resitivity

## 3. Open config file to set the initialze state of parameters.

This is the parameters that we need to set before running the estimate. For example: 
    - dataPath: Path of the input data
    - thedaD: theta
    - rho0: po
    - n_list: m
    - threshold: using for fillter data (selected sample are above the threshold)

## 4. Start estimating by running command:
`$ python main.py`

# Output:
The program will provide all the result which will include:
    - Plot of original data and fit line (.png)
    - The estimate of ro and theta (.txt)
    - The fitted value of ro (.csv)