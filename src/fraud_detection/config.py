from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class Config:
    data_path: str = os.getenv("DATA_PATH", "./data/transactions.csv")
    model_dir: str = os.getenv("MODEL_DIR", "./models")
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))


def get_config() -> Config:
    return Config()
