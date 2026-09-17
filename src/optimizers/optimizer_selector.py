import torch.optim as optim


class OptimizerSelector:

    def __init__(self, optimizer_type="adam", lr=1e-3, **kwargs):

        self.optimizer_type = optimizer_type.lower()
        self.lr = lr
        self.kwargs = kwargs

    def get_optimizer(self, model):

        if self.optimizer_type == "adam":
            return optim.Adam(
                model.parameters(),
                lr=self.lr
            )

        elif self.optimizer_type == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.kwargs.get("weight_decay", 1e-4)
            )

        elif self.optimizer_type == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=self.lr,
                momentum=self.kwargs.get("momentum", 0.9)
            )

        elif self.optimizer_type == "rmsprop":
            return optim.RMSprop(
                model.parameters(),
                lr=self.lr
            )

        else:
            raise ValueError(
                f"Optimizer '{self.optimizer_type}' not supported"
            )