from metrics.metrics import Metrics

def evaluate_model(y_pred, y_true):

    metrics_ = Metrics(y_pred, y_true)

    metrics_data = {
        "mse": metrics_.mse(),
        "mae": metrics_.mae(),
        "rmse": metrics_.rmse(),
        "r2": metrics_.r2(),
        "relative error": metrics_.relative_error(),
        "bias": metrics_.bias(), 
        "resolution": metrics_.resolution(),
        "p68": metrics_.percentile_68(),
        "p95": metrics_.percentile_95(),
        "correlation": metrics_.correlation(),
        "energy_scale": metrics_.energy_scale()
    }

    return metrics_data


def evaluate_training(y_pred, y_true):

    metrics_ = Metrics(y_pred, y_true)

    metrics_data = {
        "mse": metrics_.mse(),
        "mae": metrics_.mae(),
        "rmse": metrics_.rmse(),
        "r2": metrics_.r2(),
        "relative_error": metrics_.relative_error(),
        "bias": metrics_.bias(), 
        "resolution": metrics_.resolution(),
        "correlation": metrics_.correlation(),
        "energy_scale": metrics_.energy_scale()
    }

    return metrics_data


def training_metrics(train_loss:list=None, val_loss:list=None,
                     train_metric:list=None, val_metric:list=None):
    train_data = {
        "train_losses": train_loss,
        "val_losses": val_loss,
        "train_metric": train_metric, 
        "val_metric": val_metric
    }

    return train_data

# def model_data(**kwargs):

#     model_data = {
#         "model":,
#         "dropout":, 
#         "loss": ,
#         "n_layers":,
#         "n_neurons":,
#         "activation":,
#         "input_type":,
#         "path":,
#         "batch_size":,
#         "optimizer":,
#         "lr":,
#         "epochs":,
#         "seed":,
#         "output":,
#         "patience":
#     }

#     pass


    