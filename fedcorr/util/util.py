import dis
from math import gamma
import numpy as np
import torch
import torch.nn.functional as F
import copy
from scipy.spatial.distance import cdist
import numpy as np
from util.plots import plot_transition_heatmap, plot_label_distributions, top_label_conversions, label_transition_matrix

def sample_different_label(original_label, num_classes):
    """
    Sample a label different from original_label.
    """
    possible_labels = list(range(num_classes))
    possible_labels.remove(int(original_label))
    return np.random.choice(possible_labels)



def add_full_noise(args, y_train, dict_users, save_fig_path="./label_distribution.png"):
    np.random.seed(args.seed)
    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
    print(gamma_s)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial

    y_train_noisy = copy.deepcopy(y_train)
    real_noise_level = np.zeros(args.num_users)

    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]

        for j in noisy_idx:
            y_train_noisy[sample_idx[j]] = sample_different_label(
                y_train_noisy[sample_idx[j]],
                args.num_classes
            )

        noise_ratio = np.mean(
            np.asarray(y_train)[sample_idx] != np.asarray(y_train_noisy)[sample_idx]
        )
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio

    # Transition analysis
    M = label_transition_matrix(y_train, y_train_noisy, args.num_classes)

    plot_transition_heatmap(M,
        save_path="./label_transition_row_norm.png",
        normalize="row",
        title="Label transition (row-normalized)",
    )

    plot_transition_heatmap(
        M,
        save_path="./label_transition_counts.png",
        normalize=None,
        title="Label transition (counts)",
    )

    print("\nTop label flips:")
    for c, i, j in top_label_conversions(M, k=10):
        print(f"{c:6d} : {i} -> {j}")

    if save_fig_path is not None:
        plot_label_distributions(
            y_before=y_train,
            y_after=y_train_noisy,
            num_classes=args.num_classes,
            save_path=save_fig_path,
            title_prefix="Global label distribution",
        )

    return y_train_noisy, gamma_s, real_noise_level

def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)

    gamma_s = np.random.binomial(1, args.level_n_system, args.num_users)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial
    y_train_noisy = copy.deepcopy(y_train)
    real_noise_level = np.zeros(args.num_users)

    values, counts = np.unique(y_train, return_counts=True)
    count_dict = dict(zip(values, counts))
    print(count_dict, sum(count_dict.values()))

    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[i]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, args.num_classes, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
            i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio
    
    values, counts = np.unique(y_train_noisy, return_counts=True)
    count_dict = dict(zip(values, counts))
    print(count_dict, sum(count_dict.values()))
    return (y_train_noisy, gamma_s, real_noise_level)


def get_output(loader, net, args, latent=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device).long()

            if latent is False:
                logits = net(images)                         # [B, C]
                probs = F.softmax(logits, dim=1)             # [B, C]
            else:
                # If latent=True path exists in your model, keep as-is
                probs = net(images, True)
                logits = None

            if criterion is not None:
                if logits is None:
                    # fallback if latent=True used; not typical
                    loss = criterion(torch.log(probs + 1e-12), labels)
                else:
                    loss = criterion(logits, labels)         #  correct (logits)
            else:
                loss = None

            if i == 0:
                output_whole = probs.detach().cpu().numpy()
                loss_whole = None if loss is None else loss.detach().cpu().numpy()
            else:
                output_whole = np.concatenate((output_whole, probs.detach().cpu().numpy()), axis=0)
                if loss is not None:
                    loss_whole = np.concatenate((loss_whole, loss.detach().cpu().numpy()), axis=0)

    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole
    

def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = list(np.ogrid[:m, :n])
    idx[1] = sort_indices
    # sorted matrix
    distances_ = distances[tuple(idx)]
    # print(distances_)
    print(distances_.shape)
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids
