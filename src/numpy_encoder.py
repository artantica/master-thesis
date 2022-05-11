"""Solution if you have nested numpy arrays in a dictionary."""
import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types.

    Fix the error with NumPy array is not JSON serializable.
    """

    def default(self, obj: object) -> object:
        """Convert object to json encoder.

        :param obj: Object
        :type obj: object
        :return: Converted object
        :rtype: object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
