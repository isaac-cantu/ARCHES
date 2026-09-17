import torch.optim.lr_scheduler as lr_scheduler


class SchedulerSelector:

    def __init__(self, scheduler_type=None, **kwargs):

        self.scheduler_type = scheduler_type
        self.kwargs = kwargs

    def get_scheduler(self, optimizer):

        if self.scheduler_type == "step":

            return lr_scheduler.StepLR(
                optimizer,
                step_size=self.kwargs.get("step_size", 10),
                gamma=self.kwargs.get("gamma", 0.1)
            )

        elif self.scheduler_type == "plateau":

            return lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.kwargs.get("factor", 0.5),
                patience=self.kwargs.get("patience", 5)
            )

        elif self.scheduler_type == "cosine":

            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.kwargs.get("T_max", 50)
            )

        else:
            return None