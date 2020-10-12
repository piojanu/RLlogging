from stable_baselines import logger


class NeptuneLogger(logger.KVWriter):
    """Stable Baselines Neptune logger.

    Example usage:
    ```
    from stable_baselines import logger

    logger_ = logger.Logger.CURRENT
    logger_.output_formats.append(NeptuneLogger(...))

    ...

    ```
    """

    def __init__(self, experiment):
        super().__init__()
        self._experiment = experiment

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, 'dtype'):
                v = float(v)
            self._experiment.log_metric(k, v)
