
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = "./logs"
MODELS = ["vanilla", "fixed_step", "mlp_step"]
TIME_STEPS = 16


def read_logs(log_dir):
    data = defaultdict(list)

    with open(log_dir, "r") as log_file:
        lines = log_file.readlines()

        header, lines = lines[0], lines[1:]
        header = header.strip().split("\t")

        for line in lines:
            parts = line.strip().split("\t")
            parts = map(float, parts)

            for head, part in zip(header, parts):
                data[head].append(part)

    return data


def aggregate_data(log_path, model, time_steps):
    all_data = defaultdict(list)

    for t in range(1, time_steps):
        data = read_logs(f"{log_path}/{model}_backward_t{t}.log")

        for key, value in data.items():
            all_data[key].append(value)

    return all_data


def get_curves(data):
    low, mean, high = [], [], []

    for points in data:
        points = np.array(points)
        med = points.mean()
        l = med - points.std()
        h = med + points.std()
        mean.append(med)
        low.append(l)
        high.append(h)

    return low, mean, high


def get_combined_curves(data1, data2):
    low, mean, high = [], [], []

    for points1, point2 in zip(data1, data2):
        points1, point2 = np.array(points1), np.array(point2)
        points = points1 + point2
        med = points.mean()
        l = med - points.std()
        h = med + points.std()
        mean.append(med)
        low.append(l)
        high.append(h)

    return low, mean, high


def plot_main_gradient_times(vanilla_data, fixed_data, mlp_data, fixed_base_data=None, mlp_base_data=None):
    vanilla_low, vanilla_mean, vanilla_high = get_curves(vanilla_data)
    fixed_low, fixed_mean, fixed_high = get_curves(fixed_data)
    mlp_low, mlp_mean, mlp_high = get_curves(mlp_data)

    x = list(range(TIME_STEPS - 1))
    plt.plot(vanilla_mean, label="Vanilla", color="firebrick")
    plt.fill_between(x, vanilla_high, vanilla_mean, facecolor="lightcoral", alpha=0.6)
    plt.fill_between(x, vanilla_mean, vanilla_low, facecolor="lightcoral", alpha=0.6)

    plt.plot(fixed_mean, label="Fixed Step", color="navy")
    plt.fill_between(x, fixed_high, fixed_mean, facecolor="lightsteelblue", alpha=0.6)
    plt.fill_between(x, fixed_mean, fixed_low, facecolor="lightsteelblue", alpha=0.6)

    plt.plot(mlp_mean, label="MLP Step", color="forestgreen")
    plt.fill_between(x, mlp_high, mlp_mean, facecolor="darkseagreen", alpha=0.6)
    plt.fill_between(x, mlp_mean, mlp_low, facecolor="darkseagreen", alpha=0.6)

    if fixed_base_data is not None:
        _, fixed_base_mean, _ = get_curves(fixed_base_data)
        plt.plot(fixed_base_mean, label="Fixed Step Baseline", color="navy", linestyle="--")

    if mlp_base_data is not None:
        _, mlp_base_mean, _ = get_curves(mlp_base_data)
        plt.plot(mlp_base_mean, label="MLP Step Baseline", color="forestgreen", linestyle="--")

    plt.legend()
    plt.title("Computation time for weight gradients", fontsize=10)
    plt.tight_layout()
    plt.ylabel("time in s")
    plt.xlabel("sequence length")
    plt.savefig("./main_gradients.png")
    plt.close()


def plot_total_gradient_times(vanilla_data, fixed_data, mlp_data, fixed_base_data=None, mlp_base_data=None):
    vanilla_low, vanilla_mean, vanilla_high = get_curves(vanilla_data["main_backward"])
    fixed_low, fixed_mean, fixed_high = get_combined_curves(fixed_data["main_backward"], fixed_data["recoding_backward"])
    mlp_low, mlp_mean, mlp_high = get_combined_curves(mlp_data["main_backward"], mlp_data["recoding_backward"])

    x = list(range(TIME_STEPS - 1))
    plt.plot(vanilla_mean, label="Vanilla", color="firebrick")
    plt.fill_between(x, vanilla_high, vanilla_mean, facecolor="lightcoral", alpha=0.6)
    plt.fill_between(x, vanilla_mean, vanilla_low, facecolor="lightcoral", alpha=0.6)

    plt.plot(fixed_mean, label="Fixed Step", color="navy")
    plt.fill_between(x, fixed_high, fixed_mean, facecolor="lightsteelblue", alpha=0.6)
    plt.fill_between(x, fixed_mean, fixed_low, facecolor="lightsteelblue", alpha=0.6)

    plt.plot(mlp_mean, label="MLP Step", color="forestgreen")
    plt.fill_between(x, mlp_high, mlp_mean, facecolor="darkseagreen", alpha=0.6)
    plt.fill_between(x, mlp_mean, mlp_low, facecolor="darkseagreen", alpha=0.6)

    if fixed_base_data is not None:
        _, fixed_base_mean, _ = get_combined_curves(fixed_base_data["main_backward"], fixed_base_data["recoding_backward"])
        plt.plot(fixed_base_mean, label="Fixed Step Baseline", color="navy", linestyle="--")

    if mlp_base_data is not None:
        _, mlp_base_mean, _ = get_combined_curves(mlp_base_data["main_backward"], mlp_base_data["recoding_backward"])
        plt.plot(mlp_base_mean, label="MLP Step Baseline", color="forestgreen", linestyle="--")

    plt.legend()
    plt.title("Computation time for all gradients", fontsize=10)
    plt.tight_layout()
    plt.ylabel("time in s")
    plt.xlabel("sequence length")
    plt.savefig("./all_gradients.png")
    plt.close()


def plot_total_times(vanilla_data, fixed_data, mlp_data, fixed_base_data=None, mlp_base_data=None):
    vanilla_low, vanilla_mean, vanilla_high = get_curves(vanilla_data)
    fixed_low, fixed_mean, fixed_high = get_curves(fixed_data)
    mlp_low, mlp_mean, mlp_high = get_curves(mlp_data)

    x = list(range(TIME_STEPS - 1))
    plt.plot(vanilla_mean, label="Vanilla", color="firebrick")
    plt.fill_between(x, vanilla_high, vanilla_mean, facecolor="lightcoral", alpha=0.6)
    plt.fill_between(x, vanilla_mean, vanilla_low, facecolor="lightcoral", alpha=0.6)

    plt.plot(fixed_mean, label="Fixed Step", color="navy")
    plt.fill_between(x, fixed_high, fixed_mean, facecolor="lightsteelblue", alpha=0.6)
    plt.fill_between(x, fixed_mean, fixed_low, facecolor="lightsteelblue", alpha=0.6)

    plt.plot(mlp_mean, label="MLP Step", color="forestgreen")
    plt.fill_between(x, mlp_high, mlp_mean, facecolor="darkseagreen", alpha=0.6)
    plt.fill_between(x, mlp_mean, mlp_low, facecolor="darkseagreen", alpha=0.6)

    if fixed_base_data is not None:
        _, fixed_base_mean, _ = get_curves(fixed_base_data)
        plt.plot(fixed_base_mean, label="Fixed Step Baseline", color="navy", linestyle="--")

    if mlp_base_data is not None:
        _, mlp_base_mean, _ = get_curves(mlp_base_data)
        plt.plot(mlp_base_mean, label="MLP Step Baseline", color="forestgreen", linestyle="--")

    plt.legend()
    plt.title("Total computation time", fontsize=10)
    plt.tight_layout()
    plt.ylabel("time in s")
    plt.xlabel("sequence length")
    plt.savefig("./total.png")
    plt.close()


if __name__ == "__main__":
    # Load vanilla model data and new model baseline data
    vanilla_data = aggregate_data(LOG_PATH, "vanilla", TIME_STEPS)
    fixed_base_data = aggregate_data(LOG_PATH, "fixed_step_baseline", TIME_STEPS)
    mlp_base_data = aggregate_data(LOG_PATH, "mlp_step_baseline", TIME_STEPS)

    # Load new model dara
    fixed_data = aggregate_data(LOG_PATH, "fixed_step_single_sample", TIME_STEPS)
    mlp_data = aggregate_data(LOG_PATH, "mlp_step_single_sample", TIME_STEPS)

    # Plot
    plot_main_gradient_times(
        vanilla_data["main_backward"], fixed_data["main_backward"], mlp_data["main_backward"],
        fixed_base_data=fixed_base_data["main_backward"], mlp_base_data=mlp_base_data["main_backward"]
    )
    plot_total_gradient_times(
        vanilla_data, fixed_data, mlp_data, fixed_base_data=fixed_base_data, mlp_base_data=mlp_base_data
    )
    plot_total_times(
        vanilla_data["total"], fixed_data["total"], mlp_data["total"],
        fixed_base_data=fixed_base_data["total"], mlp_base_data=mlp_base_data["total"]
    )
