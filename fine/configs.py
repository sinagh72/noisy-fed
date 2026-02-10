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
    level_n_system: float = 0.8
    noise_p: float = 0.3
    noise_type: str = "asym"
    save_dir: str = "./results"

    # model
    model_name: str = "resnet18"
    pretrained: bool = True
    batch_size: int = 64

    # relabel
    fed_strategy: str = "fedeig"
    strategy: str = "B"     # "A" "B" "C"
    tau: float = 0.95

    strict_keep_frac: float = 0.30
    strict_min_per_class: int = 10

    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-3
    optimizer: str = "AdamW"
