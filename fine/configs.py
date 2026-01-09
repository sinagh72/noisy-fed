# configs.py
from dataclasses import dataclass

@dataclass
class ExpConfig:
    seed: int = 42
    dataset_name: str = "Srinivasan"

    iid: bool = True
    non_iid_prob_class: float = 0.0
    alpha_dirichlet: float = 0.0
    num_users: int = 1

    # noise
    level_n_system: float = 1.0
    level_n: float = 0.2
    noise_type: str = "sym"
    save_dir: str = "./figs"

    # model
    model_name: str = "resnet18"
    pretrained: bool = False
    batch_size: int = 64

    # identification
    id_mode: str = "gmm"    # "gmm" or "zeta"
    zeta: float = 0.2

    # relabel
    strategy: str = "B"     # "A" "B" "C"
    tau: float = 0.95

    # strategy B strict teacher set
    strict_keep_frac: float = 0.30
    strict_min_per_class: int = 10
