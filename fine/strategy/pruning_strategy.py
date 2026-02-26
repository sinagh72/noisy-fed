
# prunning_strategy.py
import os
from build_model import build_backbone_model
from trainer import train_model, predict_all
from fine.utils.metrics import per_class_metrics, save_metrics_report_txt
from strategy.baseline_strategy import weighted_average_state_dicts, state_dict_to_cpu

from strategy.identification import identification_process



def run_pruning_strategy(cfg, gamma_s, num_classes, base_state, train_loaders_train, test_loader, cfg_identification, cfg_train, save_dir, device):
    
    _, _, clean_cids, _ = identification_process(cfg=cfg, num_classes=num_classes, gamma_s=gamma_s,
                                                                                    train_loaders_train=train_loaders_train,
                                                                                    cfg_identification=cfg_identification, base_state=base_state,
                                                                                    save_dir=save_dir, device=device)
    global_net = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
    global_net.load_state_dict(base_state, strict=True)
    global_net.eval()

    for r in range(0, cfg.rounds):
        print(f"\n========== Round {r+1}/{cfg.rounds} ==========")
        local_states = []
        local_weights = []

        for cid in clean_cids:
            # ----- client receives global model -----
            net = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
            net.load_state_dict(global_net.state_dict(), strict=True)

            print(f"\n--- Client {cid} Training (Round {r+1}) ---")

            net, _ = train_model(
                net,
                train_loaders_train[cid],
                device,
                cfg_train,
                cfg,
                label_mode="clean",
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


    

