import matplotlib.pyplot as plt
import seaborn as sn

# individual para cada modelo
class nn_plots():

    
    def __init__(self, nn_data, path:str=None):
        pass

    # exp_nnn/model_nn/train/train.png
    def loss_plot(self):
        plt.savefig()

    # exp_nnn/model_nn/metric/metric.png
    def metric_plot(self):
        plt.savefig()

    # exp_nnn/model_nn/prediction/prediction.png
    def prediction_plot(self):
        plt.savefig()

# En grids de n x n
class exp_plots():

    def __init__(self, metadata, path:str=None):
        pass

    # exp_nnn/general/heatmap.png
    def heat_map_plot():
        plt.savefig()

    # exp_nnn/general/train/train_data.png
    def loss_map_plot():
        plt.savefig()

    # exp_nnn/general/metric/metric
    def metric_map_plot():
        plt.savefig()

    # exp_nnn/general/prediction/prediction.png
    def prediction_map_plot():
        plt.savefig()
