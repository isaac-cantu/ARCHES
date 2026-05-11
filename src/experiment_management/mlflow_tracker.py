import mlflow


class MLFlowTracker:

    def __init__(self, experiment_name):

        mlflow.set_experiment(
            experiment_name
        )

    def start_run(self, run_name):

        mlflow.start_run(
            run_name=run_name
        )

    def log_params(self, params:dict):

        mlflow.log_params(params)

    def log_metrics(self, metrics:dict):

        for key, value in metrics.items():

            if isinstance(value, list):

                if len(value) > 0:
                    mlflow.log_metric(
                        key,
                        value[-1]
                    )

            else:

                mlflow.log_metric(
                    key,
                    value
                )

    def log_artifact(self, path):

        mlflow.log_artifact(path)

    def end_run(self):

        mlflow.end_run()