import argparse
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_experiment_result(pkl_dir, pkl_filelist):
    # Collect experiment results
    pkl_lst = [l.rstrip() for l in open(pkl_filelist)]

    results_collection = dict()
    for pf in pkl_lst:
        step_size = pf.rstrip(".pkl").split("_")[-1]

        pkl_file = os.path.join(pkl_dir, pf)
        try:
            with open(pkl_file, "rb") as f:
                results = pickle.load(f)
        except OSError as e:
            print(f"OS error: {e}")
            print(f"Failed to load {pkl_file}")

        wav_files = []
        params = []
        alphas = []
        betas = []
        deltas = []
        residuals = []
        for wf in results:
            try:
                assert ("alpha" in results[wf]) and (results[wf]["alpha"])

                alpha = results[wf]["alpha"][0]
                beta = results[wf]["beta"][0]
                delta = results[wf]["delta"][0]
                r = results[wf]["Rk"][0]

                wav_files.append(wf)
                params.append([alpha, beta, delta])
                alphas.append(alpha)
                betas.append(beta)
                deltas.append(delta)
                residuals.append(r)

            except AssertionError as e:
                print(e)
                print(f"Skip {wf}")
                continue

        results_collection[pf] = {
            "step_size": step_size,
            "wav_files": wav_files,
            "params": params,
            "alphas": alphas,
            "betas": betas,
            "deltas": deltas,
            "residuals": residuals,
        }

    # Find best param settings across experiments
    num_pf = len(pkl_lst)
    num_wf = len(results)

    r_mat = np.empty((num_wf, num_pf))
    a_mat = np.empty((num_wf, num_pf))
    b_mat = np.empty((num_wf, num_pf))
    d_mat = np.empty((num_wf, num_pf))

    for k, pf in enumerate(results_collection):
        for i, wf in enumerate(results_collection[pf]["wav_files"]):
            r_mat[i, k] = results_collection[pf]["residuals"][i]
            a_mat[i, k] = results_collection[pf]["alphas"][i]
            b_mat[i, k] = results_collection[pf]["betas"][i]
            d_mat[i, k] = results_collection[pf]["deltas"][i]

    alphas_best = np.empty((num_wf,))
    betas_best = np.empty((num_wf,))
    deltas_best = np.empty((num_wf,))

    idx = np.argmin(r_mat, axis=1)
    for i, k in enumerate(idx):
        alphas_best[i] = a_mat[i, k]
        betas_best[i] = b_mat[i, k]
        deltas_best[i] = d_mat[i, k]

    return results_collection, alphas_best, betas_best, deltas_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pd",
        "--pkl_dir",
        required=True,
        help="absolute path to pkl files that store experiment results",
    )
    parser.add_argument(
        "-pf", "--pkl_filelist", nargs="+", required=True, help="list of pkl files"
    )
    args = parser.parse_args()

    (
        results_collection_norm,
        alphas_best_norm,
        betas_best_norm,
        deltas_best_norm,
    ) = get_experiment_result(args.pkl_dir, args.pkl_filelist[0])
    (
        results_collection_disorder,
        alphas_best_disorder,
        betas_best_disorder,
        deltas_best_disorder,
    ) = get_experiment_result(args.pkl_dir, args.pkl_filelist[1])

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for pf in results_collection_norm:
        ax.plot(
            np.arange(1, len(results_collection_norm[pf]["residuals"]) + 1),
            results_collection_norm[pf]["residuals"],
            "-o",
            label="step_size: " + results_collection_norm[pf]["step_size"],
        )
    ax.set_xlabel("sample id")
    ax.set_ylabel("residual")
    ax.set_title("normal")
    ax.legend()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for pf in results_collection_disorder:
        ax.plot(
            np.arange(1, len(results_collection_disorder[pf]["residuals"]) + 1),
            results_collection_disorder[pf]["residuals"],
            "-o",
            label="step_size: " + results_collection_disorder[pf]["step_size"],
        )
    ax.set_xlabel("sample id")
    ax.set_ylabel("residual")
    ax.set_title("disorder")
    ax.legend()

    fig = plt.figure()
    num_plots = len(results_collection_norm)
    num_cols = 4
    num_rows = num_plots // num_cols + 1
    for i, pf in enumerate(results_collection_norm):
        ax = fig.add_subplot(num_cols, num_rows, int(i) + 1, projection="3d")
        ax.scatter(
            results_collection_norm[pf]["alphas"],
            results_collection_norm[pf]["deltas"],
            results_collection_norm[pf]["betas"],
            marker="o",
            label="step_size: " + results_collection_norm[pf]["step_size"],
        )
        ax.set_xlabel("alpha")
        ax.set_ylabel("delta")
        ax.set_zlabel("beta")
        ax.legend()
        ax.grid(True)

    fig = plt.figure()
    num_plots = len(results_collection_disorder)
    num_cols = 4
    num_rows = num_plots // num_cols + 1
    for i, pf in enumerate(results_collection_disorder):
        ax = fig.add_subplot(num_cols, num_rows, int(i) + 1, projection="3d")
        ax.scatter(
            results_collection_disorder[pf]["alphas"],
            results_collection_disorder[pf]["deltas"],
            results_collection_disorder[pf]["betas"],
            marker="o",
            label="step_size: " + results_collection_disorder[pf]["step_size"],
        )
        ax.set_xlabel("alpha")
        ax.set_ylabel("delta")
        ax.set_zlabel("beta")
        ax.legend()
        ax.grid(True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(alphas_best_norm, deltas_best_norm, betas_best_norm, marker="o", label="norm")
    ax.scatter(
        alphas_best_disorder,
        deltas_best_disorder,
        betas_best_disorder,
        marker="o",
        label="disorder",
    )
    ax.set_xlabel("alpha")
    ax.set_ylabel("delta")
    ax.set_zlabel("beta")
    ax.legend()
    ax.grid(True)
    ax.view_init(5, -75)

    plt.tight_layout()
    plt.show()
