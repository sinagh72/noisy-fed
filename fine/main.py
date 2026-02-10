# main.py
from configs import ExpConfig
from dataset.dataset import get_dataset
from utils.set_seed import set_all_seeds
from fl import  run_n_clients
import math
import itertools
import random

def iter_noisy_sets_unique(
    num_clients: int,
    k_noisy: int,
    *,
    cases_to_run: int,
    rng: random.Random,
):
    total = math.comb(num_clients, k_noisy)
    emitted = 0

    # Case 1: enumerate all
    if total <= cases_to_run:
        for combo in itertools.combinations(range(num_clients), k_noisy):
            key = (k_noisy, combo)
            yield set(combo)
            emitted += 1
        return

    # Case 2: sample
    attempts = 0
    while emitted < cases_to_run and attempts < cases_to_run * 200:
        combo = tuple(sorted(rng.sample(range(num_clients), k_noisy)))
        key = (k_noisy, combo)
        yield set(combo)
        emitted += 1
        attempts += 1


def sweep_all_cases(
    *,
    cfg,
    dataset_train,
    dataset_test,
    dict_users,
    num_classes,
    clean_client_percentage,
    max_cases_per_percent=200,
):
    N = cfg.num_users
    seen_global = set()   # <-- GLOBAL uniqueness

    for p in clean_client_percentage:
        k = int(round(p * N))
        k = max(1, min(N - 1, k))

        total = math.comb(N, k)
        cases_to_run = min(total, max_cases_per_percent)

        rng = random.Random(cfg.seed)

        for noisy_set in iter_noisy_sets_unique(
            N,
            k,
            cases_to_run=cases_to_run,
            rng=rng,
        ):
            forced_gamma_s = [1] * N
            for idx in noisy_set:
                forced_gamma_s[idx] = 0

            run_n_clients(
                cfg=cfg,
                dataset_train=dataset_train,
                dataset_test=dataset_test,
                dict_users=dict_users,
                num_classes=num_classes,
                forced_gamma_s=forced_gamma_s,
            )

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
    sweep_all_cases(
        cfg=cfg,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        dict_users=dict_users,
        num_classes=num_classes,
        max_cases_per_percent=200,
        clean_client_percentage=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    )

    # run_n_clients(
    #             cfg=cfg,
    #             dataset_train=dataset_train,
    #             dataset_test=dataset_test,
    #             dict_users=dict_users,
    #             num_classes=num_classes,
    #             forced_gamma_s=[1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    #             )



