from losses.losses import LossSelector
from evaluation.evaluate import evaluate_training, evaluate_model
import torch
import json 

def to_serializable(obj):

    if isinstance(obj, torch.Tensor):
        return obj.item()

    return obj

class Trainer:

    def __init__(self, model, loss_criterion, optimizer, device="cpu", path:str=None):
        
        self.model = model
        self.criterion = LossSelector(loss_type=loss_criterion).get_loss()
        self.optimizer = optimizer
        self.device = device
        self.path = path

        self.history = {

            "train": {
                "loss": [],
                "mse": [],
                "mae": [],
                "rmse": [],
                "r2": [],
                "relative_error": [],
                "bias": [],
                "resolution": [],
                "correlation": [],
                "energy_scale": []
            },

            "val": {
                "loss": [],
                "mse": [],
                "mae": [],
                "rmse": [],
                "r2": [],
                "relative_error": [],
                "bias": [],
                "resolution": [],
                "correlation": [],
                "energy_scale": []
            },

            "early_stopping":{
                "best_val": float("inf"),
                "patience_counter": 0,
                "best_epoch":0
            },

            "scheduler":{
                "lr":[]
            },

            "best_epoch":{
                "epoch": 0,
                "train_loss": 0,
                "val_loss": float("inf"),
                "train_mse": 0,
                "val_mse": 0,
            }
        }


    def set_dataloaders(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def update_metrics(self, set_type, metrics_data):

        for key, value in metrics_data.items():

            self.history[set_type][key].append(value.item())    

    def train(self, epochs, early_stopping=True, patience=10):

        for epoch in range(epochs):

            # -------- TRAIN --------
            self.model.train()
            train_loss = 0

            train_preds = []
            train_targets = []

            for x, y in self.train_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                train_preds.append(pred.detach().cpu())
                train_targets.append(y.detach().cpu())

            train_loss /= len(self.train_loader)
            self.history["train"]["loss"].append(train_loss)

            train_preds = torch.cat(train_preds).squeeze()
            train_targets = torch.cat(train_targets).squeeze()
            
            self.update_metrics(set_type="train", metrics_data=evaluate_training(train_preds, train_targets))

            # -------- VALIDATION --------
            self.model.eval()
            val_loss = 0

            val_preds = []
            val_targets = []

            with torch.no_grad():
                for x, y in self.val_loader:

                    x = x.to(self.device)
                    y = y.to(self.device)

                    pred = self.model(x)
                    loss = self.criterion(pred, y)

                    val_loss += loss.item()

                    val_preds.append(pred.detach().cpu())
                    val_targets.append(y.detach().cpu())

            val_loss /= len(self.val_loader)
            self.history["val"]["loss"].append(val_loss)

            val_preds = torch.cat(val_preds).squeeze()
            val_targets = torch.cat(val_targets).squeeze()

            self.update_metrics(set_type="val", metrics_data=evaluate_training(val_preds, val_targets))

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["scheduler"]["lr"].append(current_lr)

            if ((epoch+1)%10 == 0):
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")


            else:
                pass

            # -------- EARLY STOPPING --------
            if early_stopping:

                if val_loss < self.history["early_stopping"]["best_val"]:

                    self.history["early_stopping"]["best_val"] = val_loss

                    self.history["early_stopping"]["patience_counter"] = 0

                    self.history["early_stopping"]["best_epoch"] = epoch + 1

                    # torch.save(
                    #     self.model.state_dict(),
                    #     "best_model.pth"
                    # )

                else:

                    self.history["early_stopping"]["patience_counter"] += 1

                    if (
                        self.history["early_stopping"]["patience_counter"]
                        >= patience
                    ):

                        print("Early stopping triggered")
                        break

            if val_loss < self.history["best_epoch"]["val_loss"]:

                    self.history["best_epoch"]["val_loss"] = val_loss

                    self.history["best_epoch"]["train_loss"] = train_loss

                    self.history["best_epoch"]["epoch"] = epoch + 1

                    best_model = self.model
                    
        self.model = best_model

    def get_history(self):
        return self.history

    def get_model(self):
        return self.model
    
    def save_model(self):
        model_path = self.path + "/best_model.pth"
        torch.save(self.model.state_dict(), model_path)
    
    def save_history(self):

        history_path = self.path + "/history.json"
        with open(history_path, "w") as f:

            json.dump(self.history, f, indent=4, default=to_serializable)

    def evaluate_test(self, test_loader):

        self.model.eval()

        preds = []
        targets = []

        with torch.no_grad():

            for x, y in test_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)

                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())

        preds = torch.cat(preds).squeeze()
        targets = torch.cat(targets).squeeze()

        self.test_metrics = evaluate_model(
            preds,
            targets
        )

        return preds, targets

    def save_metrics(self):

        metrics_path = self.path + "/metrics.json"

        with open(metrics_path, "w") as f:

            json.dump(self.test_metrics, f, indent=4, default=to_serializable)


