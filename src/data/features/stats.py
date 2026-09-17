import torch
import numpy as np

def stats_generator(df_particles, df_shower, device="cpu"):

    # -------- tensors --------
    x = torch.tensor(df_particles["x"].values, dtype=torch.float32, device=device)
    y = torch.tensor(df_particles["y"].values, dtype=torch.float32, device=device)
    t = torch.tensor(df_particles["t"].values, dtype=torch.float32, device=device)
    e = torch.tensor(np.log1p(df_particles["energy"].values), dtype=torch.float32, device=device)
    shower_id = torch.tensor(df_particles["shower"].values, dtype=torch.long, device=device)

    # -------- mapping --------
    unique_showers, shower_idx = torch.unique(shower_id, return_inverse=True)
    n_showers = unique_showers.shape[0]

    ones = torch.ones_like(x)

    # -------- conteo --------
    count = torch.zeros(n_showers, device=device)
    count.scatter_add_(0, shower_idx, ones)

    # -------- medias --------
    x_sum = torch.zeros(n_showers, device=device)
    y_sum = torch.zeros(n_showers, device=device)
    t_sum = torch.zeros(n_showers, device=device)
    e_sum = torch.zeros(n_showers, device=device)

    x_sum.scatter_add_(0, shower_idx, x)
    y_sum.scatter_add_(0, shower_idx, y)
    t_sum.scatter_add_(0, shower_idx, t)
    e_sum.scatter_add_(0, shower_idx, e)

    x_mean = x_sum / (count + 1e-8)
    y_mean = y_sum / (count + 1e-8)
    t_mean = t_sum / (count + 1e-8)
    e_mean = e_sum / (count + 1e-8)

    # -------- varianzas --------
    x_c = x - x_mean[shower_idx]
    y_c = y - y_mean[shower_idx]
    t_c = t - t_mean[shower_idx]
    e_c = e - e_mean[shower_idx]

    x_var = torch.zeros(n_showers, device=device)
    y_var = torch.zeros(n_showers, device=device)
    t_var = torch.zeros(n_showers, device=device)
    e_var = torch.zeros(n_showers, device=device)

    x_var.scatter_add_(0, shower_idx, x_c**2)
    y_var.scatter_add_(0, shower_idx, y_c**2)
    t_var.scatter_add_(0, shower_idx, t_c**2)
    e_var.scatter_add_(0, shower_idx, e_c**2)

    x_var /= (count + 1e-8)
    y_var /= (count + 1e-8)
    t_var /= (count + 1e-8)
    e_var /= (count + 1e-8)

    t_std = torch.sqrt(torch.clamp(t_var, min=0.0))

    # -------- skew (aproximado) --------
    t_skew = torch.zeros(n_showers, device=device)
    t_skew.scatter_add_(0, shower_idx, t_c**3)
    t_skew = t_skew / (count + 1e-8)
    t_skew = t_skew / (t_std**3 + 1e-8)

    # -------- energía --------
    e_max = torch.zeros(n_showers, device=device)
    e_max.scatter_reduce_(0, shower_idx, e, reduce="amax", include_self=False)

    # -------- covarianzas --------
    cov_xt = torch.zeros(n_showers, device=device)
    cov_yt = torch.zeros(n_showers, device=device)

    cov_xt.scatter_add_(0, shower_idx, x_c * t_c)
    cov_yt.scatter_add_(0, shower_idx, y_c * t_c)

    cov_xt /= (count + 1e-8)
    cov_yt /= (count + 1e-8)

    # -------- radio --------
    r = torch.sqrt(x_c**2 + y_c**2)

    r_sum = torch.zeros(n_showers, device=device)
    r_sq_sum = torch.zeros(n_showers, device=device)

    r_sum.scatter_add_(0, shower_idx, r)
    r_sq_sum.scatter_add_(0, shower_idx, r**2)

    r_mean = r_sum / (count + 1e-8)
    r_var = r_sq_sum/(count+1e-8) - r_mean**2

    # -------- PCA (aproximación torch) --------
    cov_xx = x_var
    cov_yy = y_var

    cov_xy = torch.zeros(n_showers, device=device)
    cov_xy.scatter_add_(0, shower_idx, x_c * y_c)
    cov_xy /= (count + 1e-8)

    # eigenvalues analíticos 2x2
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy**2

    temp = torch.sqrt(torch.clamp(trace**2 - 4*det, min=0.0))

    lambda1 = (trace + temp) / 2
    lambda2 = (trace - temp) / 2

    # ángulo
    angle = 0.5 * torch.atan2(2*cov_xy, cov_xx - cov_yy)

    # -------- final --------
    X = torch.stack([
        x_mean,
        y_mean,
        lambda1,
        lambda2,
        angle,
        t_mean,
        t_std,
        t_skew,
        e_sum,
        e_mean,
        e_var,
        e_max,
        cov_xt,
        cov_yt,
        r_mean,
        r_var
    ], dim=1)

    # -------- target alineado --------
    mapping = {val.item(): i for i, val in enumerate(unique_showers)}
    df_shower["idx"] = df_shower["event_no"].map(mapping)
    df_shower = df_shower.dropna().sort_values("idx")

    y = torch.tensor(df_shower["total_energy"].values, dtype=torch.float32, device=device)

    return X, y
