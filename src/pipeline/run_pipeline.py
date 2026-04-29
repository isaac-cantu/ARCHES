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

sys.path.append(r"/home/icantu24/Documents/ARCHES/src")

# Data
from data.conversion import convert
from data.load_db import particles_csv, shower_csv
from data.preprocessing import data_processed
from data.split_data import split_data

# Evaluation
import evaluation.plots 
import evaluation.evaluate 

# Model
import models.mlp as mlp

# Loss
from physics.losses import LossSelector

# Train
import training.metrics
import training.train
from training.trainer import Trainer

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

    for seed in search_space_config["seed"]:

        set_seed(seed)

        # Train - Test - Val - Split
        train_dataset, val_dataset, test_dataset, scaler = split_data(df_particles=input_data, df_shower=np.log1p(output_data[:,2]), 
                                                                    train=training_config["split"]["train"], 
                                                                    test=training_config["split"]["test"], 
                                                                    val=training_config["split"]["validation"], 
                                                                    seed=seed)
        
    # 2. Ajustar datos
    #print(input_data)

        for batch in training_config["batch_size"]:

            train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch,
                        shuffle=True
                        )
            
            val_loader = DataLoader(
                        val_dataset,
                        batch_size=batch,
                        shuffle=True
                        )
            
            test_loader = DataLoader(
                        test_dataset,
                        batch_size=batch,
                        shuffle=True
                    )
            
            
            for i in search_space_config["layers"]:
                for j in search_space_config["neurons"]:
                    for k in search_space_config["activation"]:
                    

                        # Model
                        model = mlp.ShowerMLP(input_dim=input_dim, output_dim=len(data_config["output"]), 
                                    n_layers=i, n_neurons=j, activation=k, dropout=model_config["dropout"])


                        criterion = LossSelector(loss_type=loss_config["type"])

                        optimizer = optim.Adam(
                            model.parameters(),
                            lr=1e-3
                        )

                        train_model = Trainer(model, loss_config["type"], optimizer)
                        train_model.set_dataloaders(train_loader, val_loader)
                        train_model.train(epochs=training_config["epochs"],
                                          early_stopping=training_config["early_stopping"]["enabled"],
                                          patience=training_config["early_stopping"]["patience"])
                        train_model.get_history()
                        model = train_model.get_model()

                        # Evaluation


                        #torch.save(model.state_dict(), f'experiments/exp_{}/model_{}/l{i}_n{j}_a{k}_b{batch}_s{seed}')

                        # Guardar Json de modelo con métricas
                        # Guardar csv de datos de entrenamiento y métricas
                        # Guardar modelo
                        # Guardar gráficas
                        
   


# (.venv) icantu24@192:~/Documents/ARCHES/src$ python3 pipeline/run_pipeline.py --config pipeline/config_example.yaml 