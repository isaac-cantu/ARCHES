from models import mlp
from trainers.trainer import Trainer
from optimizers.optimizer_selector import OptimizerSelector
from visualization.plots import TrainPlots
from experiment_management.mlflow_tracker import MLFlowTracker

class ExperimentRunner:

    def __init__(self, config, data, use_mlflow=True):

        self.config = config
        self.data = data
        self.use_mlflow = use_mlflow
        self.model_type = config["model"]["type"]

    def build_model(self):

        
        if self.model_type == "mlp":
            self.training_info = self.config["training"]
            self.model_info = self.config["model"]
            self.data_info = self.config["data"]

            self.model_name = (
                f"Model_{self.model_info['id']}: "
                f"L{self.model_info['hidden_layers']}_"
                f"W{self.model_info['neurons']}_"
                f"{self.model_info['activation']}_"
                f"S{self.training_info['seed']}_"
                f"B{self.training_info['batch_size']}_"
                f"{self.data_info['type']}"
            )

            self.model = mlp.ShowerMLP(input_dim=self.data_info["input_dim"], 
                                  output_dim=len(self.data_info["output"]), 
                                  n_layers=self.model_info["hidden_layers"], 
                                  n_neurons=self.model_info["neurons"], 
                                  activation=self.model_info["activation"], 
                                  dropout=self.training_info["dropout"],
                                  batchnorm=self.training_info["batchnorm"])

        elif self.model_type == "cnn":
            pass
        
        elif self.model_type == "kan":
            pass

        elif self.model_type == "pinn":
            pass

        else:
            pass

    def build_optimizer(self):

        self.optimizer = OptimizerSelector(optimizer_type=self.training_info["optimizer"],
                                           lr=self.training_info["lr"]
                                           ).get_optimizer(self.model)

    def build_trainer(self):
        self.train_model = Trainer(self.model, 
                              self.training_info["loss"], 
                              self.optimizer, 
                              "cpu", 
                              self.model_info["path"],
                              self.data_info["scale"])
        

    def train(self):
        self.train_model.set_dataloaders(self.data["train"], self.data["val"])
        self.train_model.train(epochs=self.training_info["epochs"],
                                early_stopping=self.training_info["early_stopping"],
                                patience=self.training_info["patience"])
        

    def evaluate(self):
        self.y_pred, self.y_true = self.train_model.evaluate_test(test_loader=self.data["test"])

    def save_results(self):

        self.train_model.save_history()
        self.train_model.save_model()
        self.train_model.save_metrics()
        self.train_model.save_model_csv(self.config)

    def plot_results(self):

        model_plot = TrainPlots(self.y_pred, 
                                self.y_true, 
                                self.model_info["path"],
                                self.model_name,
                                self.config["experiment"]["path"])
        plot_name = (
                f"Model_{self.model_info['id']}-"
                f"L{self.model_info['hidden_layers']}_"
                f"W{self.model_info['neurons']}_"
                f"{self.model_info['activation']}_"
                f"S{self.training_info['seed']}_"
                f"B{self.training_info['batch_size']}_"
                f"{self.data_info['type']}"
            )
        model_plot.plot_all(plot_name)

    def run(self):

        if self.use_mlflow:
            self.tracker = MLFlowTracker(
                experiment_name=self.config["experiment"]["name"]
            )

        self.build_model()

        if self.use_mlflow:
            self.tracker.start_run(
                run_name=self.model_name
            )

        if self.use_mlflow:

            self.tracker.log_params({

                "model_type": self.model_type,

                "hidden_layers":
                    self.model_info["hidden_layers"],

                "neurons":
                    self.model_info["neurons"],

                "activation":
                    self.model_info["activation"],

                "dropout":
                    self.training_info["dropout"],

                "batchnorm":
                    self.training_info["batchnorm"],

                "optimizer":
                    self.training_info["optimizer"],

                "lr":
                    self.training_info["lr"],

                "batch_size":
                    self.training_info["batch_size"],

                "loss":
                    self.training_info["loss"],

                "seed":
                    self.training_info["seed"]
            })
                
        self.build_optimizer()

        self.build_trainer()

        self.train()

        self.evaluate()

        if self.use_mlflow:

            metrics = self.train_model.get_history()

            self.tracker.log_metrics(
                metrics["val"]
            )

        self.save_results()

        self.plot_results()

        if self.use_mlflow:

            self.tracker.log_artifact(
                f"{self.model_info['path']}/history.json"
            )

            self.tracker.log_artifact(
                f"{self.model_info['path']}/plots/training_metrics.png"
            )

            self.tracker.log_artifact(
                f"{self.model_info['path']}/best_model.pth"
            )

        if self.use_mlflow:

            self.tracker.end_run()