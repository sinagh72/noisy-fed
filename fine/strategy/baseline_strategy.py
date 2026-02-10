# strategy.py
import torch
from build_model import build_backbone_model
from trainer import train_model, predict_all
from fine.utils.metrics import per_class_metrics, save_metrics_report_txt

def state_dict_to_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def average_state_dicts(state_dicts):
    """Simple FedAvg (equal weight)."""
    avg = {}
    for k in state_dicts[0].keys():
        vals = [sd[k] for sd in state_dicts]
        if not torch.is_floating_point(vals[0]):
            avg[k] = vals[0]  # keep as-is (or take first)
        else:
            stacked = torch.stack([v.float() for v in vals], dim=0)
            avg[k] = stacked.mean(dim=0).type_as(vals[0])
    return avg


def run_fedavg(cfg, num_classes, base_state, train_loaders_train, test_loader, cfg_train, save_dir, device):
        print(f"\n===  FedAvg Training ({cfg.num_rounds} rounds) ===")
        # ----- initialize global model -----
        global_net = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
        global_net.load_state_dict(base_state, strict=True)
        global_state = state_dict_to_cpu(global_net.state_dict())

        for r in range(cfg.num_rounds):
            print(f"\n========== Round {r+1}/{cfg.num_rounds} ==========")

            local_states = []

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

            # ----- FedAvg aggregation -----
            global_state = average_state_dicts(local_states)

            # optional: quick round-level evaluation
            global_net.load_state_dict(global_state, strict=True)
            global_net.eval()

            yt, yp = predict_all(global_net, test_loader, device)
            m = per_class_metrics(yt, yp, num_classes=num_classes)

            save_metrics_report_txt(
                save_dir,
                "TEST performance: net_final (FedAvg, multi-round, noisy clients)",
                m,
                num_classes,
                extra_lines=[
                    f"\n========== Round {r+1} ==========",
                ],
            )