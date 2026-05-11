# Libraries
import yaml
import torch
import torch.optim as optim
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
from visualization.plots import TrainPlots
from evaluation.evaluate import evaluate_model


# Model
import models.mlp as mlp
from experiments.experiment_runner import ExperimentRunner

# Loss
from losses.losses import LossSelector

# Train
import metrics.metrics
import trainers.train
from trainers.trainer import Trainer
from visualization.data_plots import DataPlots
from visualization.exp_plots import ExpPlots

# Pipeline
from pipeline.set_seed import set_seed
from pipeline.utils import read_parser, read_yaml, model_information
#==========================================
# Lectura de argumento parse --arg

yaml_path, verbose, gpu_activation = read_parser()

#==========================================
# Abrir archivo .yaml
config = read_yaml(yaml_path=yaml_path)

#pprint(config)

experiment_config      = config["experiment"]
data_config            = config["data"]
model_config           = config["model"]
training_config        = config["training"]
search_space_config    = config["search_space"]
loss_config            = config["loss"]
metrics_config         = config["metrics"]

#==========================================

if __name__ == "__main__":

# Data
    #==========================================

    print(data_config["path"])

    # 1. Cargar datos
    #==========================================
    df_particles = particles_csv(data_config["path"])
    df_shower = shower_csv(data_config["path"])

    input_data, output_data = data_processed(input_type=data_config["input_type"], df_particles=df_particles, df_shower=df_shower)
    #input_data, output_data = convert(data)

    #==========================================

    input_dim = len(input_data[0])

    #==========================================
    path = experiment_config["path"]
    exp_file = path + f"/exp_{experiment_config["id"]}" 

    DataPlots(df_particles, df_shower, exp_file).plot_all()
    
    Path(exp_file).mkdir(exist_ok=True)

    with open(exp_file + "/config.yaml", "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False
        )
    # Archivo .md de resumen
    #==========================================

    for seed in search_space_config["seed"]:
        model_id = 0
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
            
            training_data = {
                "train": train_loader,
                "val": val_loader,
                "test": test_loader,
            }
            
            for layers in search_space_config["layers"]:
                for neurons in search_space_config["neurons"]:
                    for activation in search_space_config["activation"]:
                        
                        model_id += 1
                        model_params = {
                            "id": model_id,
                            "seed": seed,
                            "batch": batch,
                            "layers": layers,
                            "neurons": neurons,
                            "activation": activation,
                            "input_dim": input_dim,
                            "output": data_config["output"]
                        }

                        model_data = model_information(config, model_params, exp_file)
                       
                       #========================================== (pantalla agregar en train)
                        model_name = f"Model {model_id}: L{layers}, W{neurons}, {activation}, S{seed}, B{batch}, {data_config["input_type"]}"
                        print(model_name)

                        # Model
                        #==========================================
                        
                        ExperimentRunner(model_data, training_data).run()
                        #==========================================

                        ExpPlots(exp_file).plot_all()
                        

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

print("Experiment completed!")
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
   

# Me falta gráfica completa
# summary.csv (general)
# models.json
# plots/
# losses (PAFL) agregar
# arreglar trainer

# Probar


# (.venv) icantu24@192:~/Documents/ARCHES/src$ python3 pipeline/run_pipeline.py --config pipeline/config_example.yaml 