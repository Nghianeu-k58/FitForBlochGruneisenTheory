import yaml

from scipy.optimize import minimize

from enums import *
from utils import *


DEMO_DATA_PATH = "./data/demo.csv"
CONFIG_FILE = "./configure.yaml"

def main():
    
    # loading data from config file:
    
    print(f"Loading config from config file at: {CONFIG_FILE}")
    with open(CONFIG_FILE, mode="r") as f:
        config = yaml.safe_load(f)    
    

    # Extract configuration
    print("Extract config parameters ...")
    
    # paths
    data_path = str(config[Fileds.paths][Paths.dataPath])
    output_path = str(config[Fileds.paths][Paths.outputPath])

    # parameters
    thetaD_bounds = config[Fileds.parameters][Parameters.thedaD]

    theta_upper_bound = int(thetaD_bounds[Bounds.upperBound])
    theta_lower_bound = int(thetaD_bounds[Bounds.lowerBound])

    rho0_bounds = config[Fileds.parameters][Parameters.rho0]
    rho0_upper_bound = int(rho0_bounds[Bounds.upperBound])
    rho0_lower_bound = int(rho0_bounds[Bounds.lowerBound])
    
    n_list = [config[Fileds.parameters][Parameters.n_list]]  # ensure it's a list
    threshold = int(config[Fileds.parameters][Parameters.threshold])
    
    # minimize paramters
    methods = list(config[Fileds.minimizeparameters][MinimizeParameters.method])
    max_iter = int(config[Fileds.minimizeparameters][MinimizeParameters.fit_iteration])
    disp = bool(config[Fileds.minimizeparameters][MinimizeParameters.displayProcess])


    # Loading data
    print(f"Loading data at {data_path}")
    sub_data = read_sub_data(data_path, 0)
    fit_data = filter_rows_by_threshold(sub_data, 0, threshold)

    # create parameters

    bounds = [
        (rho0_lower_bound, rho0_upper_bound),
        (theta_lower_bound, theta_upper_bound),
        ] + [(0, None)] * len(n_list)

    p0 = [
        (rho0_upper_bound + rho0_lower_bound) // 2,
        (theta_upper_bound + theta_lower_bound) // 2,
        1e-5,
    ]
    
    for method in methods:
        
        # Optimize to find po and thetaD
        try:
            result = minimize(
                fit_residual, 
                p0, 
                args=(
                    fit_data[:, 0], 
                    fit_data[:, 1], 
                    n_list,
                ),
                method=method,
                bounds=bounds,
                options={'maxiter': max_iter, 'disp': disp}
            )
        except ValueError:
            print(f"Method {method} is not suitable for this data.")
            continue

        # export data
        rho0, theta_D, rho_n = result.x

        print(f"Optimized parameters with method {method}:")
        print(f"rho0: {rho0}")
        print(f"theta_D: {theta_D}")
        for _, n in enumerate(n_list):
            print(f"rho_{n}: {rho_n}")

        # Preparing content for text file
        content = create_content(
            method=method,
            rho0=rho0,
            thetaD=theta_D,
            rho_n=rho_n,
        )
        output_text_path = create_output_text_path(data_path, method)

        # write text file
        with open(output_text_path, mode="w", encoding="utf-8") as f:
            for line in content:
                f.write(line)

        # preparing for export image
        image_path = create_output_images_path(input_path=data_path, method=method)
        
        rho_fit = resistivity_model(sub_data[:, 0], rho0, theta_D, [rho_n], n_list)
        export_images_to_files(
            output_path=image_path, 
            original_data=sub_data, 
            fit_data=rho_fit,
        )

        # preparing csv file
        fitted_data_path = create_fitted_data_path(input_path=data_path, method=method)
        export_fitted_data(path=fitted_data_path, x=sub_data[:, 0].tolist(), y_pred=rho_fit)
        

    print("Completed!")


if __name__ == "__main__":
    main()