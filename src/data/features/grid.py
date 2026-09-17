import torch
import numpy as np

def grid_generator(df_particles, df_shower, n=8, device="cpu", flatten=True, x_range=None, y_range=None):
    """
    x_range / y_range: (min, max) tuples for the spatial grid extent.

    IMPORTANT for batched/streaming use (see data.build_feature_dataset):
    if left as None, the grid bin edges default to this call's own
    df_particles["x"]/["y"].min()/.max() -- fine for a single call over
    the whole dataset, but WRONG if you call this once per batch while
    streaming a huge file, since each batch would then define spatially
    different grid cells (batch A's cell (0,0) would cover a different
    physical region than batch B's cell (0,0)), silently corrupting the
    resulting features. Pass a fixed x_range/y_range (e.g. from
    data.build_feature_dataset.compute_global_xy_range()) whenever you are
    calling this function more than once over different subsets of the
    same dataset.
    """

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
    x_min, x_max = (x.min(), x.max()) if x_range is None else x_range
    y_min, y_max = (y.min(), y.max()) if y_range is None else y_range
    x_bins = torch.linspace(x_min, x_max, n+1, device=device)
    y_bins = torch.linspace(y_min, y_max, n+1, device=device)

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
    if flatten:
        # Unchanged from before: (n_showers, n, n, 6) -> flat vector, for
        # the MLP path. Kept exactly as-is so existing MLP+grid results are
        # reproducible (the CNN path below uses a different, channel-first
        # layout and is only taken when flatten=False).
        X = torch.stack([count, e_sum, e_mean, e_std, t_mean, t_std], dim=-1)
        X = X.view(n_showers, -1)
    else:
        # Channel-first (n_showers, 6, n, n), as expected by
        # models.cnn.ShowerCNN / torch.nn.Conv2d.
        X = torch.stack([count, e_sum, e_mean, e_std, t_mean, t_std], dim=1)

    # target
    mapping = {val.item(): i for i, val in enumerate(unique_showers)}
    df_shower["idx"] = df_shower["event_no"].map(mapping)
    df_shower = df_shower.dropna().sort_values("idx")

    y = torch.tensor(df_shower["total_energy"].values, dtype=torch.float32, device=device)

    return X, y
