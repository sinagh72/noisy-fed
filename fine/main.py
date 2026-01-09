# main.py
from configs import ExpConfig
from dataset import get_dataset
from pipeline import set_all_seeds, run_all_clients

if __name__ == "__main__":
    cfg = ExpConfig()
    set_all_seeds(cfg.seed)

    dataset_train, dataset_test, dict_users, num_classes = get_dataset(
        dataset_name=cfg.dataset_name,
        num_users=cfg.num_users,
        iid=cfg.iid,
        non_iid_prob_class=cfg.non_iid_prob_class,
        alpha_dirichlet=cfg.alpha_dirichlet,
        seed=cfg.seed,
    )

    run_all_clients(
        cfg=cfg,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        dict_users=dict_users,
        num_classes=num_classes,
    )
