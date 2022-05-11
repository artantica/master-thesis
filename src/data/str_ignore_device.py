"""Ignore device on string.

This is a hack class.

"""


class StrIgnoreDevice(str):
    """Changes `to` method to do nothing.

    The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device: str) -> object:
        """Replace changing device on `task_name` value.

        :param device: the device on which a torch.Tensor is or will be allocated
        :type device: str
        :return: self
        :rtype: object
        """
        return self
