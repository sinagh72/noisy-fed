import matplotlib.pyplot as plt


def plot_matrix(input_matrix, num_classes, plot_name, save_dir, size=(7,6), vmin=0.0, vmax=1.0):
    plt.figure(figsize=size)
    im = plt.imshow(input_matrix, vmin=vmin, vmax=vmax)
    plt.colorbar(im)

    plt.title("Inter-class Separation Matrix (Principal Directions)")
    plt.xlabel("Class j")
    plt.ylabel("Class i")

    plt.xticks(range(num_classes))
    plt.yticks(range(num_classes))

    # # annotate each cell
    for i in range(num_classes):
        for j in range(num_classes):
            value = input_matrix[i, j]
            plt.text(
                j, i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9
            )

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{plot_name}.png")
    plt.close()
