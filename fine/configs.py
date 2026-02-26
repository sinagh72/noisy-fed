# configs.py
from dataclasses import dataclass

@dataclass
class ExpConfig:
    seed: int = 42
    dataset_name: str = "cifar10"

    iid: bool = True
    non_iid_prob_class: float = 1.0
    alpha_dirichlet: float = 0.1
    num_users: int = 10

    # noise
    level_n_system: float = 0.9
    noise_p: float = 0.3
    noise_type: str = "sym"
    save_dir: str = "./results"

    # model
    model_name: str = "resnet18"
    pretrained: bool = True
    batch_size: int = 64
    
    # cutmix
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0   # 0.5–1.0 common
    cutmix_p: float = 0.5       # 0.3–0.7 common

    # FL
    fl_strategy: str = "fedeig"
    rounds: int = 100

    # relabel
    identification_epochs: int = 5
    cleaning_rounds = 10

    # training param
    epochs: int = 1
    lr: float = 5e-5
    weight_decay: float = 1e-2
    optimizer: str = "AdamW"
