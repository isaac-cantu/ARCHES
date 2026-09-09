#normalización
#limpieza
#selección de features
import torch 

from data.features.grid import grid_generator
from data.features.circle import circle_generator
from data.features.stats import stats_generator

def data_processed(input_type:str=None, df_particles=None, df_shower=None):

    input_type = input_type.lower()

    if input_type == "grid":
        n = 8
        input_data, output_data = grid_generator(n=n, df_particles=df_particles, df_shower=df_shower)
        #input_data = input_data.reshape(input_data.shape[0],6*n**2)
        print(input_data.shape, output_data.shape)
    elif input_type == "circle":
        input_data, output_data = circle_generator(df_particles=df_particles, df_shower=df_shower)

    elif input_type == "stats":
        input_data, output_data = stats_generator(df_particles=df_particles, df_shower=df_shower)
    
    else:
        print("No se ingreso un input correcto, [grid, circles, stats]")


    return input_data, output_data

def normalize_data():
    return

