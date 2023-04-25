class AvgMeter:
    def __init__(self, name="Metric"):
        """
        Creates an instance of AvgMeter class.

        Parameters:
        name (str): A string indicating the name of the metric. Default value is "Metric".
        """
        self.name = name
        self.reset()

    def reset(self):
        """
        Resets the metric values to 0.
        """
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        """
        Updates the metric with new values.

        Parameters:
        val (float): A float indicating the new value of the metric.
        count (int): An integer indicating the number of values to update. Default value is 1.
        """
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        """
        Returns a string representation of the metric value.

        Returns:
        str: A string representation of the metric value.
        """
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    """
    Returns the learning rate of the optimizer.

    Parameters:
    optimizer (torch.optim.Optimizer): A PyTorch optimizer.

    Returns:
    float: The learning rate of the optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
