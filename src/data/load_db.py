# Carga datasets (raw, processed)

import pandas as pd
#import polaris -> futura mejora para mayor velocidad


def DAT_file(path: str = None):
    """Read a raw CORSIKA .DAT file directly, skipping the CSV step.

    Requires the `corsario` package (pip install corsario; see
    https://github.com/isaac-cantu/corsario). Returns (df_particles,
    df_shower) with the exact same column names as particles_csv()/
    shower_csv(), so it is a drop-in alternative to converting to CSV
    first, e.g.:

        df_particles, df_shower = DAT_file("run/DAT000001")

    instead of:

        # once, offline:  corsario-convert run/DAT000001 particles.csv
        df_particles = particles_csv("run/processed/example")
        df_shower    = shower_csv("run/processed/example")
    """
    try:
        import corsario
    except ImportError as exc:
        raise ImportError(
            "DAT_file() requires the 'corsario' package. Install it with: "
            "pip install corsario  (see https://github.com/isaac-cantu/corsario)"
        ) from exc

    run = corsario.open(path)
    df_particles = run.to_dataframe("particles")
    df_shower = run.to_dataframe("showers")
    return df_particles, df_shower


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