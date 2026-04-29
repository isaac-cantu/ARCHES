from physics.losses import LossSelector
import torch

class Trainer:

    def __init__(self, model, loss_criterion, optimizer, device="cpu"):
        
        self.model = model
        self.criterion = LossSelector(loss_type=loss_criterion).get_loss()
        self.optimizer = optimizer
        self.device = device

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "best_val": float("inf"),
            "patience_counter": 0
        }

    def set_dataloaders(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, epochs, early_stopping=True, patience=10):

        for epoch in range(epochs):

            # -------- TRAIN --------
            self.model.train()
            train_loss = 0

            for x, y in self.train_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            self.history["train_loss"].append(train_loss)

            # -------- VALIDATION --------
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for x, y in self.val_loader:

                    x = x.to(self.device)
                    y = y.to(self.device)

                    pred = self.model(x)
                    loss = self.criterion(pred, y)

                    val_loss += loss.item()

            val_loss /= len(self.val_loader)
            self.history["val_loss"].append(val_loss)

            print(f"Epoch {epoch+1} | Train:{train_loss:.4f} | Val:{val_loss:.4f}")

            # -------- EARLY STOPPING --------
            if early_stopping:

                if val_loss < self.history["best_val"]:
                    self.history["best_val"] = val_loss
                    self.history["patience_counter"] = 0

                    torch.save(self.model.state_dict(), "best_model.pth")

                else:
                    self.history["patience_counter"] += 1

                    if self.history["patience_counter"] >= patience:
                        print("Early stopping triggered")
                        break

    def get_history(self):
        return self.history

    def get_model(self):
        return self.model