import argparse
import yaml
from pathlib import Path
import os 

def read_parser():
    parser = argparse.ArgumentParser(description="Configuración del experimento")

    # 1. Usar 'required=True' para el archivo de config
    parser.add_argument('-c', '--configs', type=str, required=True,
                        help="Ruta al archivo YAML del experimento")

    # 2. Para booleanos, lo ideal es usar 'action'
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Muestra datos detallados del modelo (default: False)")

    parser.add_argument('-g', '--gpu', action='store_true',
                        help="Activa el uso de GPU (default: False)")

    parser.add_argument('-o', '--optuna', action='store_true',
                        help="Activa el uso de OPTUNA (default: False)")

    parser.add_argument('-m', '--mlflow', action='store_true',
                        help="Activa el uso de MLFLOW (default: False)")


    args = parser.parse_args()

    # Acceso directo a los atributos
    return args.configs, args.verbose, args.gpu


def read_yaml(yaml_path:str=None):

    # Abrir archivo .yaml
    with open(yaml_path, "r") as file:
        config: dict = yaml.safe_load(file)
        #pprint(config, sort_dicts=False)

    return config

def model_information(config, model_params, exp_path):

    model_folder = exp_path + f"/model_{model_params["id"]}"
    Path(model_folder).mkdir(exist_ok=True)

    model_path = model_folder + f"/seed_{model_params["seed"]}"
    Path(model_path).mkdir(exist_ok=True)
    
    experiment_config      = config["experiment"]
    data_config            = config["data"]
    model_config           = config["model"]
    training_config        = config["training"]
    search_space_config    = config["search_space"]
    loss_config            = config["loss"]
    metrics_config         = config["metrics"]

    model_data = {
        "experiment": {
            "id": experiment_config["id"],
            "name": experiment_config["name"],
            "description": experiment_config["description"],
            "path": exp_path
        },

        "model": {
            "id": model_params["id"],
            "type": model_config["type"],
            "hidden_layers": model_params["layers"],
            "neurons": model_params["neurons"],
            "activation": model_params["activation"],
            "path": model_path 
        },

        "training": {
            "batch_size": model_params["batch"],
            "epochs": training_config["epochs"],
            "early_stopping": training_config["early_stopping"]["enabled"],
            "patience": training_config["early_stopping"]["patience"],
            "dropout": model_config["dropout"],
            "loss": loss_config["type"],
            "batchnorm": model_config["batchnorm"],
            "optimizer": training_config["optimizer"]["type"],
            "lr": float(training_config["optimizer"]["lr"]),
            "scheduler_enabled": training_config["scheduler"]["enabled"],
            "scheduler_type": training_config["scheduler"]["type"],
            "seed": model_params["seed"]
        },

        "data": {
            "type": data_config["input_type"],
            "output": data_config["output"],
            "input_dim": model_params["input_dim"],
            "scale": data_config["scale"]
        }
    }

    if os.path.isfile(model_folder+"/config.yaml"):
        pass
    else:
        with open(model_folder+"/config.yaml", "w") as f:
            general_data = {

                "model": {
                    "id": model_params["id"],
                    "type": model_config["type"],
                    "hidden_layers": model_params["layers"],
                    "neurons": model_params["neurons"],
                    "activation": model_params["activation"],
                    "path":model_folder
                },

                "training": {
                    "batch_size": model_params["batch"],
                    "epochs": training_config["epochs"],
                    "early_stopping": training_config["early_stopping"]["enabled"],
                    "patience": training_config["early_stopping"]["patience"],
                    "dropout": model_config["dropout"],
                    "loss": loss_config["type"],
                    "batchnorm": model_config["batchnorm"],
                    "optimizer": training_config["optimizer"]["type"],
                    "lr": float(training_config["optimizer"]["lr"])
                },

                "data": {
                    "type": data_config["input_type"],
                    "output": data_config["output"],
                    "input_dim": model_params["input_dim"],
                    "scale": data_config["scale"]
                }
            }

            yaml.dump(
                general_data,
                f,
                default_flow_style=False
            )


    with open(model_path+"/config.yaml", "w") as f:
        yaml.dump(
            model_params,
            f,
            default_flow_style=False
        )

    return model_data

if __name__ == "__main__":
    pass