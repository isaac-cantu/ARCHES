import torch
import numpy as np

def circle_generator(df_particles, df_shower, device="cpu"):

    # -------- tensors --------
    x = torch.tensor(df_particles["x"].values, dtype=torch.float32, device=device)
    y = torch.tensor(df_particles["y"].values, dtype=torch.float32, device=device)
    e = torch.tensor(np.log1p(df_particles["energy"].values), dtype=torch.float32, device=device)
    t = torch.tensor(df_particles["t"].values, dtype=torch.float32, device=device)
    shower_id = torch.tensor(df_particles["shower"].values, dtype=torch.long, device=device)

    # -------- mapping --------
    unique_showers, shower_idx = torch.unique(shower_id, return_inverse=True)
    n_showers = unique_showers.shape[0]

    # -------- medias por shower --------
    count = torch.zeros(n_showers, device=device)
    x_sum = torch.zeros(n_showers, device=device)
    y_sum = torch.zeros(n_showers, device=device)

    ones = torch.ones_like(x)

    count.scatter_add_(0, shower_idx, ones)
    x_sum.scatter_add_(0, shower_idx, x)
    y_sum.scatter_add_(0, shower_idx, y)

    x_mean = x_sum / (count + 1e-8)
    y_mean = y_sum / (count + 1e-8)

    # -------- centrar --------
    x_c = x - x_mean[shower_idx]
    y_c = y - y_mean[shower_idx]

    r = torch.sqrt(x_c**2 + y_c**2)

    # -------- sigma radial --------
    r_sum = torch.zeros(n_showers, device=device)
    r_sq_sum = torch.zeros(n_showers, device=device)

    r_sum.scatter_add_(0, shower_idx, r)
    r_sq_sum.scatter_add_(0, shower_idx, r**2)

    r_mean = r_sum / (count + 1e-8)
    r_var = r_sq_sum/(count+1e-8) - r_mean**2
    sigma = torch.sqrt(torch.clamp(r_var, min=0.0))

    # -------- bins --------
    r_bins = torch.stack([sigma * i for i in range(1,5)], dim=1)  # (n_showers, 4)

    # -------- asignar anillos --------
    # expandir para comparar
    r_exp = r.unsqueeze(1)  # (N_particles, 1)
    bins = r_bins[shower_idx]  # (N_particles, 4)

    ring_idx = torch.zeros_like(r, dtype=torch.long)

    ring_idx += (r > bins[:,0])
    ring_idx += (r > bins[:,1])
    ring_idx += (r > bins[:,2])
    ring_idx += (r > bins[:,3])

    ring_idx = torch.clamp(ring_idx, max=3)

    # -------- índice combinado (shower + ring) --------
    flat_idx = shower_idx * 4 + ring_idx
    size = n_showers * 4

    # -------- acumulaciones --------
    count_r = torch.zeros(size, device=device)
    e_sum = torch.zeros(size, device=device)
    e_sq_sum = torch.zeros(size, device=device)
    t_sum = torch.zeros(size, device=device)
    t_sq_sum = torch.zeros(size, device=device)

    count_r.scatter_add_(0, flat_idx, ones)
    e_sum.scatter_add_(0, flat_idx, e)
    e_sq_sum.scatter_add_(0, flat_idx, e**2)
    t_sum.scatter_add_(0, flat_idx, t)
    t_sq_sum.scatter_add_(0, flat_idx, t**2)

    # -------- reshape --------
    count_r = count_r.view(n_showers, 4)
    e_sum = e_sum.view(n_showers, 4)
    t_sum = t_sum.view(n_showers, 4)

    # -------- medias --------
    mask = count_r > 0

    e_mean = torch.zeros_like(e_sum)
    t_mean = torch.zeros_like(t_sum)

    e_mean[mask] = e_sum[mask] / count_r[mask]
    t_mean[mask] = t_sum[mask] / count_r[mask]

    # -------- std --------
    e_var = e_sq_sum.view(n_showers,4)/(count_r+1e-8) - e_mean**2
    t_var = t_sq_sum.view(n_showers,4)/(count_r+1e-8) - t_mean**2

    e_std = torch.sqrt(torch.clamp(e_var, min=0.0))
    t_std = torch.sqrt(torch.clamp(t_var, min=0.0))

    # -------- features --------
    features = torch.stack([
        count_r,
        e_sum,
        e_mean,
        e_std,
        t_mean,
        t_std
    ], dim=2)  # (n_showers, 4, 6)

    features = features.view(n_showers, -1)  # (n_showers, 24)

    # -------- covarianza --------
    xx = torch.zeros(n_showers, device=device)
    yy = torch.zeros(n_showers, device=device)
    xy = torch.zeros(n_showers, device=device)

    xx.scatter_add_(0, shower_idx, x_c**2)
    yy.scatter_add_(0, shower_idx, y_c**2)
    xy.scatter_add_(0, shower_idx, x_c*y_c)

    cov_xx = xx / (count + 1e-8)
    cov_yy = yy / (count + 1e-8)
    cov_xy = xy / (count + 1e-8)

    cov = torch.stack([cov_xx, cov_xy, cov_xy, cov_yy], dim=1)

    # -------- final --------
    X = torch.cat([
        x_mean.unsqueeze(1),
        y_mean.unsqueeze(1),
        cov,
        features
    ], dim=1)

    # -------- target --------
    mapping = {val.item(): i for i, val in enumerate(unique_showers)}
    df_shower["idx"] = df_shower["event_no"].map(mapping)
    df_shower = df_shower.dropna().sort_values("idx")

    y = torch.tensor(df_shower["total_energy"].values, dtype=torch.float32, device=device)

    return X, y
