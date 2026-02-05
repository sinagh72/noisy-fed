# configs.py
from dataclasses import dataclass

@dataclass
class ExpConfig:
    seed: int = 42
    dataset_name: str = "cifar10"

    iid: bool = True
    non_iid_prob_class: float = 0.0
    alpha_dirichlet: float = 0.0
    num_users: int = 10

    # noise
    level_n_system: float = 1.0
    level_n: float = 0.4
    noise_type: str = "asym"
    save_dir: str = "./results"

    # model
    model_name: str = "resnet18"
    pretrained: bool = True
    batch_size: int = 64

    # identification
    id_mode: str = "gmm"    # "gmm" or "zeta"
    zeta: float = 0.2

    # relabel
    strategy: str = "B"     # "A" "B" "C"
    tau: float = 0.95


    strict_keep_frac: float = 0.30
    strict_min_per_class: int = 10

    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-2
    optimizer: str = "AdamW"
