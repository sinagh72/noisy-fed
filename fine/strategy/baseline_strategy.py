# base_strategy.py
import os
import torch
from build_model import build_backbone_model
from trainer import train_model, predict_all
from fine.utils.metrics import per_class_metrics, save_metrics_report_txt

def state_dict_to_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def weighted_average_state_dicts(state_dicts, weights):
    """
    Args:
        state_dicts: List[Dict[str, Tensor]]
        weights:     List[int or float] (same length)

    Returns:
        averaged state_dict
    """
    assert len(state_dicts) == len(weights)
    total_weight = float(sum(weights))
    assert total_weight > 0

    avg = {}
    for k in state_dicts[0].keys():
        if not torch.is_floating_point(state_dicts[0][k]):
            avg[k] = state_dicts[0][k]
            continue

        avg[k] = sum(
            (w / total_weight) * sd[k]
            for sd, w in zip(state_dicts, weights)
        )

    return avg


def run_fedavg(cfg, gamma_s, num_classes, base_state, train_loaders_train, test_loader, cfg_train, save_dir, device):
        print(f"\n==================  FedAvg Training ({cfg.rounds} rounds) ==================")
        # ----- initialize global model -----
        global_net = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
        global_net.load_state_dict(base_state, strict=True)
        global_state = state_dict_to_cpu(global_net.state_dict())

        for r in range(cfg.rounds):
            print(f"\n========== Round {r+1}/{cfg.rounds} ==========")

            local_states = []
            local_weights = []

            for cid in range(cfg.num_users):
                # ----- client receives global model -----
                net = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
                net.load_state_dict(global_state, strict=True)

                print(f"\n--- Client {cid} Training (Round {r+1}) ---")

                net, _ = train_model(
                    net,
                    train_loaders_train[cid],
                    None,
                    device,
                    cfg_train,
                    label_mode="noisy",
                    num_classes=num_classes,
                )

                local_states.append(state_dict_to_cpu(net.state_dict()))
                local_weights.append(len(train_loaders_train[cid].dataset))

            # ----- FedAvg aggregation -----
            global_state = weighted_average_state_dicts(local_states, local_weights)

            # optional: quick round-level evaluation
            global_net.load_state_dict(global_state, strict=True)
            global_net.eval()

            yt, yp, loss = predict_all(global_net, test_loader, device)
            m = per_class_metrics(yt, yp, num_classes=num_classes)
            m["loss"] = loss

            save_metrics_report_txt(
                os.path.join(save_dir, f"results.txt"),
                "TEST performance: net_final",
                m,
                num_classes,
                extra_lines=[
                    f"\n================ Round {r+1} ================",
                    f"Clients: {gamma_s}"
                ],
            )