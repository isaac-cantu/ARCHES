# Carga datasets (raw, processed)

import pandas as pd
#import polaris -> futura mejora para mayor velocidad

def DAT_file(path:str = None):
    return

def particles_csv(path:str = None):
    particles_path = path+"/particles.csv"
    df = pd.read_csv(particles_path)
    return df

def shower_csv(path:str = None):
    shower_path = path+"/shower.csv" 
    df = pd.read_csv(shower_path)
    return df

if __name__ == "__main__":
    #path = "/home/icantu24/Documents/ARCHES/data/processed/example";
    #print(particles_csv(path).head())
    pass