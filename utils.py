"""
Define all helper functions.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def read_sub_data(path, start_idx):
    """
    Read csv file and return 2D numpy array.
    
    Input: 
        path (string): path of the data file.
        start_idx (int): start index column 
    Output:
        2D numpy array with shape (rows, 3)
    """
    sub_data = [[] for _ in range(2)]
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.split(",")
            
            for i in range(2):
                sub_data[i].append(parts[start_idx + i])
    return np.array(sub_data, dtype=np.float64).T

def debye_integral(n, td_over_t):
    """Numerical integral for a given exponent n and TD/T ratio"""
    integrand = lambda x: x**n / ((np.exp(x) - 1) * (1 - np.exp(-x)))
    value, _ = quad(integrand, 0, td_over_t)
    return value


def resistivity_model(T, rho0, theta_D, rho_n_list, n_list):
    """Full generalized resistivity model"""
    T = np.asarray(T)
    result = np.full_like(T, rho0, dtype=float)

    for rho_n, n in zip(rho_n_list, n_list):
        scale = (T / theta_D)**n
        integral_values = np.array([debye_integral(n, theta_D / t) for t in T])
        result += (n - 1) * rho_n * scale * integral_values
    return result


def fit_residual(params, T, rho_exp, n_list):
    rho0 = params[0]
    theta_D = params[1]
    rho_n_list = params[2:]
    
    rho_model = resistivity_model(T, rho0, theta_D, rho_n_list, n_list)
    return np.sum((rho_model - rho_exp)**2)


def filter_rows_by_threshold(data, column_index, threshold):
    """
    Fillter data using threshold in column k.

    Input:
        data (numpy array)
        column_index (int) column that apply filtering
        threshold (int) threshold for filtering
    """
    col = data[:, column_index]
    # Find rows that are NOT outliers in the selected column
    mask = (col >= threshold)
    # Return filtered data
    return data[mask]


def create_content(method, rho0, thetaD, rho_n):
    content = []
    content.append(f"Optimized parameters with method {method}:\n")
    content.append(f"Value of rho_0: {rho0}\n")
    content.append(f"Value of thetaD: {thetaD}\n")
    content.append(f"rho_4: {rho_n}:\n")
    return content

def create_output_text_content(
    methods: list,
    roh0s: list,
    thetaDs: list,
    rho_n: list,
):
    """
    Create and return content in file.
    """

    contents = []
    contents.append("#"*30)

    for i in range(len(methods)):
        tmp_data = create_content(
            method=methods[i],
            roh0s=roh0s[i],
            thetaDs=thetaDs[i],
            rho_n=rho_n[i],
        )
        contents += tmp_data
        contents.append("#"*30)

    return contents  

def create_output_text_path(input_path, method):
    """
    Create and return output path for text file.
    """

    file_name = os.path.basename(input_path).split(".")[0]
    
    return os.path.join(os.getcwd(), "output", "texts", f"output_of_{file_name}_method_{method}.txt")


def create_output_images_path(input_path, method):
    """
    Create and return output path for text file.
    """

    file_name = os.path.basename(input_path).split(".")[0]
    
    return os.path.join(os.getcwd(), "output", "images", f"output_of_{file_name}_method_{method}.png")


def create_fitted_data_path(input_path, method):
    """
    Create and reutrn output of fitted data path.
    """
    file_name = os.path.basename(input_path).split(".")[0]
    return os.path.join(os.getcwd(), "output", "fitted", f"output_of_{file_name}_method_{method}.csv")


def export_images_to_files(output_path, original_data, fit_data):
    _, ax = plt.subplots(figsize=(8,8))
    ax.plot(original_data[:, 0], original_data[:, 1], 'o', label='Experimental', markersize=2)
    ax.plot(original_data[:, 0], fit_data, '-', label='Fit')
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Resistivity (Ohm·m)")
    ax.legend()
    ax.figure.savefig(output_path, dpi=400)
    print(f"Save image to {output_path} successful.")

def export_fitted_data(path, x, y_pred):
    """
    Export data in fitted data.
    """

    tmp_arr = np.array([x, y_pred])
    np.savetxt(path, tmp_arr.T, delimiter=",")
    print(f"Export fitted file successful at {path}")