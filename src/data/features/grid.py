import torch
import numpy as np

def grid_generator(df_particles, df_shower, n=8, device="cpu"):

    # convertir a torch 
    x = torch.tensor(df_particles["x"].values, dtype=torch.float32, device=device)
    y = torch.tensor(df_particles["y"].values, dtype=torch.float32, device=device)
    e = torch.tensor(np.log1p(df_particles["energy"]).values, dtype=torch.float32, device=device)
    t = torch.tensor(df_particles["t"].values, dtype=torch.float32, device=device)
    shower_id = torch.tensor(df_particles["shower"].values, dtype=torch.long, device=device)

    # mapear showers a índices 
    showers = df_particles.shower.unique()
    unique_showers, shower_idx = torch.unique(shower_id, return_inverse=True)
    n_showers = unique_showers.shape[0]

    # bins 
    x_bins = torch.linspace(x.min(), x.max(), n+1, device=device)
    y_bins = torch.linspace(y.min(), y.max(), n+1, device=device)

    ix = torch.bucketize(x, x_bins) - 1
    iy = torch.bucketize(y, y_bins) - 1

    # máscara válida 
    valid = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)

    ix = ix[valid]
    iy = iy[valid]
    e = e[valid]
    t = t[valid]
    shower_idx = shower_idx[valid]

    # índice global 
    flat_idx = shower_idx * (n*n) + ix * n + iy

    # acumulaciones 
    size = n_showers * n * n

    count = torch.zeros(size, device=device)
    e_sum = torch.zeros(size, device=device)
    e_sq_sum = torch.zeros(size, device=device)
    t_sum = torch.zeros(size, device=device)
    t_sq_sum = torch.zeros(size, device=device)

    ones = torch.ones_like(e)

    count.scatter_add_(0, flat_idx, ones)
    e_sum.scatter_add_(0, flat_idx, e)
    e_sq_sum.scatter_add_(0, flat_idx, e**2)
    t_sum.scatter_add_(0, flat_idx, t)
    t_sq_sum.scatter_add_(0, flat_idx, t**2)

    # reshape
    count = count.view(n_showers, n, n)
    e_sum = e_sum.view(n_showers, n, n)

    # medias
    eps = 1e-8
    mask = count > 0

    e_mean = torch.zeros_like(e_sum)
    t_mean = torch.zeros_like(t_sum.view(n_showers, n, n))

    e_mean[mask] = e_sum[mask] / count[mask]
    t_mean[mask] = t_sum.view(n_showers, n, n)[mask] / count[mask]

    # varianzas
    e_var = e_sq_sum.view(n_showers, n, n)/(count+eps) - e_mean**2
    t_var = t_sq_sum.view(n_showers, n, n)/(count+eps) - t_mean**2

    # std seguras
    e_std = torch.sqrt(torch.clamp(e_var, min=0.0))
    t_std = torch.sqrt(torch.clamp(t_var, min=0.0))

    # stack final
    X = torch.stack([count, e_sum, e_mean, e_std, t_mean, t_std], dim=-1)

    # flatten 
    X = X.view(n_showers, -1)

    # target
    mapping = {val.item(): i for i, val in enumerate(unique_showers)}
    df_shower["idx"] = df_shower["event_no"].map(mapping)
    df_shower = df_shower.dropna().sort_values("idx")

    y = torch.tensor(df_shower["total_energy"].values, dtype=torch.float32, device=device)

    return X, y
