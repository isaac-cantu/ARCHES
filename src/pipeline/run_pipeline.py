# Libraries
import yaml
import torch
import torch.optim as optim
import argparse
from pprint import pprint
import datetime
import sys
import math
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path

sys.path.append(r"/home/icantu24/Documents/ARCHES/src")

# Data
from data.conversion import convert
from data.load_db import particles_csv, shower_csv
from data.preprocessing import data_processed
from data.split_data import split_data

# Evaluation
from evaluation.plots import TrainPlots
from evaluation.evaluate import evaluate_model

# Model
import models.mlp as mlp

# Loss
from losses.losses import LossSelector

# Train
import metrics.metrics
import trainers.train
from trainers.trainer import Trainer

# Pipeline
from pipeline.set_seed import set_seed

# Lectura de argumento parse --arg

parser = argparse.ArgumentParser()      # Crear parser

# Argumetos/opciones
parser.add_argument('-c', '--configs', type=str, 
                    nargs=1, help="File of the experiment")

parser.add_argument('-v', '--verbose', type=bool, 
                    nargs=1, help="Show model data")

parser.add_argument('-g', '--gpu', type=bool, 
                    nargs=1, help="Show model data")



args = parser.parse_args()              # Argumento escrito

yaml_path = args.configs[0]             # Obtener argumento

#verbose = args.verbose[0]

exit
# Abrir archivo .yaml
with open(yaml_path, "r") as file:
    config: dict = yaml.safe_load(file)
    #pprint(config, sort_dicts=False)

experiment_config      = config["experiment"]
data_config            = config["data"]
model_config           = config["model"]
training_config        = config["training"]
search_space_config    = config["search_space"]
loss_config            = config["loss"]
metrics_config         = config["metrics"]

n = 10
if __name__ == "__main__":

# Data
    print(data_config["path"])
    # 1. Cargar datos
    df_particles = particles_csv(data_config["path"])
    df_shower = shower_csv(data_config["path"])

    input_data, output_data = data_processed(input_type=data_config["input_type"], df_particles=df_particles, df_shower=df_shower)
    #input_data, output_data = convert(data)

    input_dim = len(input_data[0])

    model_n = 0
    path = experiment_config["path"]
    exp_file = path + f"/exp_{experiment_config["id"]}" 

    Path(exp_file).mkdir(exist_ok=True)

    with open(exp_file + "/config.yaml", "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False
        )

    # Carpeta de experimento 
    # Archivo yaml duplicado
    # Archivo .md de resumen

    for seed in search_space_config["seed"]:

        set_seed(seed)

        # Train - Test - Val - Split
        train_dataset, val_dataset, test_dataset = split_data(X=input_data, y=output_data, 
                                                                    train=training_config["split"]["train"], 
                                                                    val=training_config["split"]["validation"]) #agregar seed
        
    # 2. Ajustar datos
    #print(input_data)

        for batch in training_config["batch_size"]:

            train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
            val_loader   = DataLoader(val_dataset, batch_size=batch)
            test_loader  = DataLoader(test_dataset, batch_size=batch)
            
            
            for i in search_space_config["layers"]:
                for j in search_space_config["neurons"]:
                    for k in search_space_config["activation"]:
                        
                        model_n += 1
                        model_path = exp_file + f"/model_{model_n}"
                        Path(model_path).mkdir(exist_ok=True)

                        model_data = {

                            "model": {
                                "type": model_config["type"],
                                "hidden_layers": i,
                                "activation": k
                            },

                            "training": {
                                "batch_size": batch,
                                "epochs": training_config["epochs"],
                                "dropout": 0,
                                "loss": 0,
                                "batchnorm": 0,
                                "optimizer": training_config["optimizer"]["type"],
                                "lr": training_config["optimizer"]["lr"],
                                "seed":0
                            }
                        }

                        with open(model_path+"/config.yaml", "w") as f:
                            yaml.dump(
                                model_config,
                                f,
                                default_flow_style=False
                            )

                        # Crear carpeta de modelo (LN_WN_k_batch_datatype)
                        # Guardar (history.json, history.csv, plots/, predictions.json, metrics.json, model.pth, metadata.json)
                        #

                        # Hacer modelo
                       
                        print(f"Model {model_n}: L{i}, W{j}, {k}, {seed}, {batch}, {data_config["input_type"]}")
                        # Model
                        model = mlp.ShowerMLP(input_dim=input_dim, output_dim=len(data_config["output"]), 
                                    n_layers=i, n_neurons=j, activation=k, dropout=True) #model_config["dropout"]


                        criterion = LossSelector(loss_type=loss_config["type"])

                        optimizer = optim.Adam(
                            model.parameters(),
                            lr=1e-3
                        )

                        train_model = Trainer(model, loss_config["type"], optimizer, "cpu", model_path)
                        train_model.set_dataloaders(train_loader, val_loader)
                        train_model.train(epochs=training_config["epochs"],
                                          early_stopping=training_config["early_stopping"]["enabled"],
                                          patience=training_config["early_stopping"]["patience"])
                        train_model.save_history()
                        train_model.save_model()
                        y_pred, y_true = train_model.evaluate_test(test_loader=test_loader)
                        train_model.save_metrics()

                        model_plot = TrainPlots(y_pred, y_true, model_path)
                        model_plot.plot_all()
                        
                        # Evaluation
                        # evaluate()
                        # obtener resultados con métricas 

                        # torch.save(model.state_dict(), f'experiments/exp_{}/model_{}/l{i}_n{j}_a{k}_b{batch}_s{seed}')

                        # Guardar Json de modelo con métricas generales
                        # Guardar csv de datos de entrenamiento y métricas
                        # Guardar modelo
                        # Guardar gráficas

                        # agregar datos a general y exp

                        # Agregar a ML flow
#     model_001/
# ├── config.yaml
# ├── best_model.pth
# ├── history.json
# ├── metrics.json
# ├── predictions.npy 
# ├── summary.csv ---------------------> pasar al general 
# └── plots/
                


# general
# summary.csv
# model.json
# plots/
# .md 

    # Gráficas generales
    # Datos generales
    # Resumen de modelos 
    # Orden de mejor a peor modelo
    # 
   

Me falta gráfica completa
summary.csv
models.json
plots/
losses (PAFL)
arreglar trainer

Probar


# (.venv) icantu24@192:~/Documents/ARCHES/src$ python3 pipeline/run_pipeline.py --config pipeline/config_example.yaml 