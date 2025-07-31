"""
Define all enums class
"""

class Fileds:
    parameters = "parameters"
    minimizeparameters = "minimizeparameters"
    paths = "paths"

class Bounds:
    lowerBound = "lowerBound"
    upperBound = "upperBound"

class Parameters:
    thedaD = "thedaD"
    rho0 = "rho0"
    n_list = "n_list"
    threshold = "threshold"

    def __str__(self):
        return "parameters"

class MinimizeParameters:

    fit_iteration = "fit_iteration"
    displayProcess = "displayProcess"
    method = "method"

    def __str__(self):
        return "minimizeParameters"
    
class Paths:

    dataPath = "dataPath"
    outputPath = "outputPath"

    def __str__(self):
        return "paths"