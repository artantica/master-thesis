"""Config file."""
import os

HOME_DIR = os.getcwd()
MODELS_DIR = "models"
DATA_FORMAT = "%d%m%Y%H%M"
RESULTS_FILE = "test_json.csv"

columns = ["input_ids", "attention_mask", "labels"]

models_names = {
    "deberta": "microsoft/deberta-base",
    "structbert": "bayartsogt/structbert-large",
    "ernie": "nghuyong/ernie-2.0-en",
}

labels_dict = {
    "meta_zero": 0,
    "meta_minus_m": 1,
    "meta_plus_m": 2,
    "meta_amb": 3,
    "z_zero": 0,
    "z_minus_m": 1,
    "z_plus_m": 2,
    "z_amb": 3,
}
