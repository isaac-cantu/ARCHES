from models import mlp
from trainers.trainer import Trainer
from optimizers.optimizer_selector import OptimizerSelector
from optimizers.scheduler_selector import SchedulerSelector
from visualization.plots import TrainPlots
from experiment_management.mlflow_tracker import MLFlowTracker

class ExperimentRunner:

    def __init__(self, config, data, use_mlflow=True, epoch_callback=None):

        self.config = config
        self.data = data
        self.use_mlflow = use_mlflow
        self.model_type = config["model"]["type"]
        # Optional per-epoch hook (epoch:int, val_loss:float) -> None,
        # forwarded to Trainer.train(); may raise to stop training early
        # (see search/run_search.py's strategy: optuna pruning support).
        self.epoch_callback = epoch_callback

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
            self.training_info = self.config["training"]
            self.model_info = self.config["model"]
            self.data_info = self.config["data"]

            self.model_name = (
                f"Model_{self.model_info['id']}: "
                f"CNN_B{self.model_info['hidden_layers']}_"
                f"C{self.model_info['neurons']}_"
                f"{self.model_info['activation']}_"
                f"S{self.training_info['seed']}_"
                f"B{self.training_info['batch_size']}_"
                f"{self.data_info['type']}"
            )

            from models.cnn import ShowerCNN

            self.model = ShowerCNN(
                in_channels=self.data_info["in_channels"],
                output_dim=len(self.data_info["output"]),
                n_blocks=self.model_info["hidden_layers"],
                base_channels=self.model_info["neurons"],
                activation=self.model_info["activation"],
                dropout=self.training_info["dropout"],
                batchnorm=self.training_info["batchnorm"],
            )
        
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

    def build_scheduler(self):
        # `training.scheduler.enabled`/`type` were already read into
        # scheduler_enabled/scheduler_type by model_information(), but
        # nothing ever built or used a scheduler from them until now.
        if self.training_info.get("scheduler_enabled"):
            self.scheduler = SchedulerSelector(
                scheduler_type=self.training_info.get("scheduler_type")
            ).get_scheduler(self.optimizer)
        else:
            self.scheduler = None

    def train(self):
        self.train_model.set_dataloaders(self.data["train"], self.data["val"])
        self.train_model.train(epochs=self.training_info["epochs"],
                                early_stopping=self.training_info["early_stopping"],
                                patience=self.training_info["patience"],
                                scheduler=self.scheduler,
                                epoch_callback=self.epoch_callback)
        

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

        self.build_scheduler()

        self.build_trainer()

        self.train()

        self.evaluate()

        if self.use_mlflow:

            metrics = self.train_model.get_history()

            self.tracker.log_metrics(
                metrics["val"]
            )

        self.save_results()

        # A degenerate/diverged run (exploding loss, dead units, or -- as
        # surfaced during testing -- a near-constant target) can produce
        # NaN/zero-variance residuals that crash matplotlib's histogram
        # binning deep inside plot_all(). That should not throw away an
        # otherwise-valid trained model + metrics (already saved above),
        # especially during an unattended multi-run search/sweep.
        try:
            self.plot_results()
        except Exception as exc:
            print(f"[ExperimentRunner] plot_results() failed for "
                  f"{getattr(self, 'model_name', '?')}, continuing without "
                  f"plots: {type(exc).__name__}: {exc}")

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