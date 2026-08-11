import gc
import os
import sys
import time
import traceback
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import faiss
import numpy as np
import optuna
import scipy.sparse as sp
from scipy.sparse import csr_matrix, identity as sp_identity
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml, make_blobs
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
from torchvision import datasets as torchvision_datasets

warnings.filterwarnings("ignore", category=UserWarning)


def save_run_to_xml(
    root,
    D,
    dt,
    graph_mean,
    graph_std,
    init_mean,
    init_std,
    rd_mean,
    rd_std,
    ari_mean,
    ari_std,
    nmi_mean,
    nmi_std,
):
    run = ET.SubElement(root, "run")

    params = ET.SubElement(run, "parameters")
    ET.SubElement(params, "D").text = str(D)
    ET.SubElement(params, "dt").text = str(dt)

    graph = ET.SubElement(run, "graph_generation")
    ET.SubElement(graph, "mean").text = str(graph_mean)
    ET.SubElement(graph, "std").text = str(graph_std)

    init_u_elem = ET.SubElement(run, "init_u")
    ET.SubElement(init_u_elem, "mean").text = str(init_mean)
    ET.SubElement(init_u_elem, "std").text = str(init_std)

    rd = ET.SubElement(run, "reaction_diffusion")
    ET.SubElement(rd, "mean").text = str(rd_mean)
    ET.SubElement(rd, "std").text = str(rd_std)

    metrics = ET.SubElement(run, "metrics")

    ari = ET.SubElement(metrics, "ARI")
    ET.SubElement(ari, "mean").text = str(ari_mean)
    ET.SubElement(ari, "std").text = str(ari_std)

    nmi = ET.SubElement(metrics, "NMI")
    ET.SubElement(nmi, "mean").text = str(nmi_mean)
    ET.SubElement(nmi, "std").text = str(nmi_std)


def _compute_gpu_knn_chunk(
    gpu_id, gpu_rows_indices, X_normalized, k, row_chunk, col_chunk, n_samples
):
    device = torch.device(f"cuda:{gpu_id}")
    local_rows = []
    local_cols = []

    with torch.cuda.device(device):
        for start_r in range(0, len(gpu_rows_indices), row_chunk):
            end_r = min(start_r + row_chunk, len(gpu_rows_indices))
            current_rows = gpu_rows_indices[start_r:end_r]

            X_chunk_gpu = torch.from_numpy(X_normalized[current_rows]).to(device)

            combined_vals = torch.full(
                (len(current_rows), k + 1), -float("inf"), device=device
            )
            combined_cols = torch.zeros(
                (len(current_rows), k + 1), dtype=torch.long, device=device
            )

            for start_c in range(0, n_samples, col_chunk):
                end_c = min(start_c + col_chunk, n_samples)

                X_col_gpu = torch.from_numpy(X_normalized[start_c:end_c]).to(device)
                sim_chunk = torch.mm(X_chunk_gpu, X_col_gpu.t())

                local_vals, local_indices = torch.topk(
                    sim_chunk, min(k + 1, sim_chunk.shape[1]), dim=1, largest=True
                )
                actual_global_cols = local_indices + start_c

                merged_vals = torch.cat([combined_vals, local_vals], dim=1)
                merged_cols = torch.cat([combined_cols, actual_global_cols], dim=1)

                top_vals, top_idx = torch.topk(
                    merged_vals, k + 1, dim=1, largest=True
                )
                combined_vals = top_vals
                combined_cols = torch.gather(merged_cols, 1, top_idx)

                del X_col_gpu, sim_chunk, local_vals, local_indices

            cols_neighbors = combined_cols[:, 1:].cpu().numpy().ravel()
            rows_neighbors = np.repeat(current_rows, k)

            local_rows.append(rows_neighbors)
            local_cols.append(cols_neighbors)

            del X_chunk_gpu, combined_vals, combined_cols
            torch.cuda.empty_cache()

    if len(local_rows) > 0:
        return np.concatenate(local_rows), np.concatenate(local_cols)
    return np.array([], dtype=np.int64), np.array([], dtype=np.int64)


def build_knn_affinity_graph(X, k, row_chunk=20000, col_chunk=15000, gpus=[0, 1, 2, 3]):
    n_samples = X.shape[0]

    if not isinstance(X, np.ndarray):
        X = X.toarray() if hasattr(X, "toarray") else np.array(X)

    X_tensor = torch.from_numpy(X).float()
    X_norms = torch.norm(X_tensor, p=2, dim=1, keepdim=True).clamp(min=1e-8)
    X_normalized = (X_tensor / X_norms).numpy()

    num_gpus = len(gpus)
    chunks_per_gpu = np.array_split(np.arange(n_samples), num_gpus)

    all_rows, all_cols = [], []

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_idx, gpu_id in enumerate(gpus):
            gpu_rows_indices = chunks_per_gpu[gpu_idx]
            if len(gpu_rows_indices) == 0:
                continue

            f = executor.submit(
                _compute_gpu_knn_chunk,
                gpu_id,
                gpu_rows_indices,
                X_normalized,
                k,
                row_chunk,
                col_chunk,
                n_samples,
            )
            futures.append(f)

        for f in futures:
            rows, cols = f.result()
            if len(rows) > 0:
                all_rows.append(rows)
                all_cols.append(cols)

    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    final_vals = np.ones_like(final_cols, dtype=np.float32)

    W_raw = csr_matrix((final_vals, (final_rows, final_cols)), shape=(n_samples, n_samples))

    W_mutual = W_raw.multiply(W_raw.T)
    W_mutual.eliminate_zeros()
    W_mutual.data = np.ones_like(W_mutual.data, dtype=np.float32)

    W_squared = W_mutual.dot(W_mutual)
    W_squared.eliminate_zeros()

    W_lil = W_squared.tolil()
    for i in range(n_samples):
        row_indices = W_lil.rows[i]
        row_data = W_lil.data[i]

        if len(row_indices) > k:
            top_k_idx = np.argpartition(row_data, -k)[-k:]
            W_lil.rows[i] = [row_indices[idx] for idx in top_k_idx]
            W_lil.data[i] = [row_data[idx] for idx in top_k_idx]

    W_mutual = W_lil.tocsr()
    W_mutual.data = np.ones_like(W_mutual.data, dtype=np.float32)

    del W_squared, W_lil
    return W_mutual


def _cg_worker_gpu(gpu_id, data, A_sparse_gpu, maxiter, tol_sq, stream):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    if stream is None:
        stream = torch.cuda.current_stream(device=device)

    with torch.cuda.device(device), torch.cuda.stream(stream):
        X_local = data["X"]
        R_local = data["R"]
        P_local = data["P"]
        Rs_old_local = data["Rs_old"]

        eps = torch.tensor(1e-12, device=device)

        for _ in range(maxiter):
            AP_local = A_sparse_gpu @ P_local

            P_AP_local = torch.sum(P_local * AP_local, dim=0)
            P_AP_stable = torch.where(P_AP_local <= 1e-12, eps, P_AP_local)
            alpha_local = Rs_old_local / P_AP_stable

            X_local += P_local * alpha_local
            R_local -= AP_local * alpha_local

            Rs_new_local = torch.sum(R_local * R_local, dim=0)

            if torch.all(Rs_new_local < tol_sq):
                break

            Rs_old_stable = torch.where(Rs_old_local <= 1e-12, eps, Rs_old_local)
            beta_local = Rs_new_local / Rs_old_stable

            P_local = R_local + P_local * beta_local
            Rs_old_local = Rs_new_local

        stream.synchronize()
        return X_local


def parallel_multi_gpu_block_cg(A_cpu, B_cpu, maxiter=200, tol=1e-6, gpus=[0, 1, 2, 3]):
    n, n_rhs = B_cpu.shape
    num_gpus = len(gpus)

    A_sparse_gpus = {}
    A_coo = A_cpu.tocoo()
    indices = torch.from_numpy(np.vstack((A_coo.row, A_coo.col))).long()
    values = torch.from_numpy(A_coo.data).float()
    A_torch_all = (
        torch.sparse_coo_tensor(indices, values, size=A_cpu.shape)
        .coalesce()
        .to_sparse_csr()
    )

    for gpu_id in gpus:
        device = torch.device(f"cuda:{gpu_id}")
        A_sparse_gpus[gpu_id] = A_torch_all.to(device)

    del A_coo, indices, values, A_torch_all
    torch.cuda.empty_cache()

    cols_per_gpu = int(np.ceil(n_rhs / num_gpus))
    gpu_data = {}

    for idx_gpu, gpu_id in enumerate(gpus):
        device = torch.device(f"cuda:{gpu_id}")
        c_start = idx_gpu * cols_per_gpu
        c_end = min(c_start + cols_per_gpu, n_rhs)

        if c_start >= n_rhs:
            break

        with torch.cuda.device(device):
            B_local = torch.from_numpy(B_cpu[:, c_start:c_end]).to(device, dtype=torch.float32)
            X_local = torch.zeros_like(B_local)
            R_local = B_local.clone()
            P_local = R_local.clone()
            Rs_old_local = torch.sum(R_local * R_local, dim=0)

            gpu_data[gpu_id] = {
                "device": device,
                "c_start": c_start,
                "c_end": c_end,
                "X": X_local,
                "R": R_local,
                "P": P_local,
                "Rs_old": Rs_old_local,
            }
            del B_local

    tol_sq = tol**2

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_id in range(len(gpu_data)):
            chunk = gpu_data[gpu_id]
            f = executor.submit(
                _cg_worker_gpu,
                gpu_id,
                chunk,
                A_sparse_gpus[gpu_id],
                maxiter,
                tol_sq,
                None,
            )
            futures.append(f)

        results = [f.result() for f in futures]

    X_final_cpu = np.zeros((n, n_rhs), dtype=np.float32)
    for idx, gpu_id in enumerate(gpu_data.keys()):
        data = gpu_data[gpu_id]
        c_start = data["c_start"]
        c_end = data["c_end"]

        X_final_cpu[:, c_start:c_end] = results[idx].cpu().numpy()
        del data["X"], data["R"], data["P"], data["Rs_old"]

    del results, futures, A_sparse_gpus
    torch.cuda.empty_cache()

    return X_final_cpu


def compute_unnormalized_laplacian(W):
    d = np.array(W.sum(axis=1)).flatten()
    D = csr_matrix((d, (np.arange(len(d)), np.arange(len(d)))), shape=W.shape)
    return D - W


def compute_symmetric_laplacian(W, device_id=0):
    device = torch.device(f"cuda:{device_id}")
    n = W.shape[0]

    if isinstance(W, np.ndarray):
        W_torch = torch.from_numpy(W).float().to(device)
        d = torch.sum(W_torch, dim=1)
        d_inv_sqrt = torch.zeros_like(d)
        mask = d > 0
        d_inv_sqrt[mask] = 1.0 / torch.sqrt(d[mask])

        D_inv_sqrt_mat = torch.diag(d_inv_sqrt)
        I_torch = torch.eye(n, device=device)
        L_sym_torch = I_torch - torch.mm(torch.mm(D_inv_sqrt_mat, W_torch), D_inv_sqrt_mat)

        return L_sym_torch.cpu().numpy()
    else:
        W_coo = W.tocoo()
        rows = torch.from_numpy(W_coo.row).long().to(device)
        cols = torch.from_numpy(W_coo.col).long().to(device)
        vals = torch.from_numpy(W_coo.data).float().to(device)
        indices = torch.stack([rows, cols])

        d = torch.bincount(rows, weights=vals, minlength=n)
        d_inv_sqrt = torch.zeros_like(d)
        mask = d > 0
        d_inv_sqrt[mask] = 1.0 / torch.sqrt(d[mask])

        norm_vals = vals * d_inv_sqrt[rows] * d_inv_sqrt[cols]
        norm_W_torch = torch.sparse_coo_tensor(indices, norm_vals, size=W.shape, device=device).coalesce()

        eye_indices = torch.arange(n, device=device).unsqueeze(0).repeat(2, 1)
        I_torch = torch.sparse_coo_tensor(eye_indices, torch.ones(n, device=device), size=W.shape, device=device)

        L_torch = (I_torch - norm_W_torch).coalesce()

        return csr_matrix(
            (
                L_torch.values().cpu().numpy(),
                (L_torch.indices()[0].cpu().numpy(), L_torch.indices()[1].cpu().numpy()),
            ),
            shape=W.shape,
        )


def compute_random_walk_laplacian(W, device_id=2):
    device = torch.device(f"cuda:{device_id}")
    n = W.shape[0]

    W_coo = W.tocoo()
    rows = torch.from_numpy(W_coo.row).long().to(device)
    cols = torch.from_numpy(W_coo.col).long().to(device)
    vals = torch.from_numpy(W_coo.data).float().to(device)
    indices = torch.stack([rows, cols])

    d = torch.bincount(rows, weights=vals, minlength=n)
    d_inv = torch.zeros_like(d)
    mask = d > 0
    d_inv[mask] = 1.0 / d[mask]

    rw_vals = vals * d_inv[rows]
    norm_W_torch = torch.sparse_coo_tensor(indices, rw_vals, size=W.shape, device=device).coalesce()

    eye_indices = torch.arange(n, device=device).unsqueeze(0).repeat(2, 1)
    I_torch = torch.sparse_coo_tensor(eye_indices, torch.ones(n, device=device), size=W.shape, device=device)

    L_rw_torch = (I_torch - norm_W_torch).coalesce()

    return csr_matrix(
        (
            L_rw_torch.values().cpu().numpy(),
            (L_rw_torch.indices()[0].cpu().numpy(), L_rw_torch.indices()[1].cpu().numpy()),
        ),
        shape=W.shape,
    )


def run_pytorch_kmeans(x, k, max_iter=100, device="cuda:0"):
    centroids = x[torch.randperm(x.shape[0])[:k]]

    for _ in range(max_iter):
        distances = (
            torch.sum(x**2, dim=1, keepdim=True)
            + torch.sum(centroids**2, dim=1)
            - 2 * torch.mm(x, centroids.T)
        )
        labels = torch.argmin(distances, dim=1)
        new_centroids = torch.stack([x[labels == i].mean(dim=0) for i in range(k)])

        if torch.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


def initialize_membership_matrix(X, C=3, scheme="spectral_soft", L=None, tau=1.0, main_gpu=0):
    n = X.shape[0]
    device = torch.device(f"cuda:{main_gpu}")

    if scheme == "kmeans_soft":
        km = KMeans(n_clusters=C, n_init=10).fit(X)
        d2 = km.transform(X)
        U = torch.from_numpy(np.exp(-d2 / tau)).to(device)

    elif scheme == "spectral_soft":
        L_coo = L.tocoo()
        indices = torch.from_numpy(np.vstack((L_coo.row, L_coo.col))).long()
        values = torch.from_numpy(L_coo.data).float()

        L_torch_sparse = torch.sparse_coo_tensor(
            indices, values, size=L.shape, device=device
        ).coalesce().to_sparse_csr()

        X_guess = torch.randn(n, C, device=device, dtype=torch.float32)

        eigenvalues, V2 = torch.lobpcg(
            L_torch_sparse, k=C, X=X_guess, largest=False, tol=1e-5, niter=200
        )

        _, labels = run_pytorch_kmeans(V2, C, device=device)

        U = torch.eye(C, device=device)[labels]
        U += 0.05 * torch.rand(n, C, device=device)

    elif scheme == "dirichlet":
        U_np = np.random.dirichlet(np.ones(C) * 0.1, size=n)
        U = torch.from_numpy(U_np).to(device, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown initialization scheme: {scheme}")

    U /= U.sum(dim=1, keepdim=True)
    return U


def compute_reaction_force(u, beta=20.0, alpha=0.5):
    return beta * u * (1 - u) * (u - alpha)


def run_imex_reaction_diffusion(
    X_cpu,
    k=22,
    D=0.2,
    dt=0.1,
    K=15,
    C=3,
    beta=100.0,
    alpha=0.5,
    gpus=[0],
):
    t0 = time.time()
    W = build_knn_affinity_graph(X_cpu, k, 20000, 15000, gpus=gpus)
    L = compute_random_walk_laplacian(W)

    L = L + sp.diags(np.full(L.shape[0], 1e-5), format="csr")

    n = X_cpu.shape[0]
    I = sp_identity(n, format="csr")
    time_graph = (time.time() - t0) * 1000

    t_init_start = time.time()
    U = initialize_membership_matrix(X_cpu, C, scheme="spectral_soft", L=L, main_gpu=gpus[0])
    time_init = (time.time() - t_init_start) * 1000

    main_device = torch.device(f"cuda:{gpus[0]}")
    t_solver_start = time.time()
    A_sys = I + dt * D * L

    with torch.cuda.device(main_device):
        for _ in range(K):
            F = compute_reaction_force(U, beta, alpha)
            B_gpu = U + dt * F
            B_cpu = B_gpu.cpu().numpy()

            U_new_cpu = parallel_multi_gpu_block_cg(A_sys, B_cpu, gpus=gpus)

            U_tensor = torch.from_numpy(U_new_cpu).to(main_device)
            max_idx = torch.argmax(U_tensor, dim=1)

            U_projected = torch.eye(C, device=main_device)[max_idx]

            lambda_param = 0.10
            U_tensor = (1.0 - lambda_param) * U_projected + lambda_param * U
            U_tensor /= U_tensor.sum(dim=1, keepdim=True)

    time_solver = (time.time() - t_solver_start) * 1000
    y_pred = torch.argmax(U, dim=1).cpu().numpy()

    return y_pred, time_graph, time_init, time_solver


def generate_sparse_synthetic_data(
    n_samples=100000, centers=5, n_features=2, sparsity=0.5, random_state=42
):
    X_raw, y_raw = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        random_state=random_state,
    )
    mask = np.random.random(X_raw.shape) > sparsity
    X_sparse = X_raw * mask
    row_has_data = np.any(X_sparse != 0, axis=1)

    X_filtered = X_sparse[row_has_data]
    y_filtered = y_raw[row_has_data]
    return csr_matrix(X_filtered), y_filtered


def generate_dense_synthetic_data(
    n_samples=100000, centers=5, n_features=2, random_state=42
):
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        random_state=random_state,
    )
    return csr_matrix(X), y_true


def optuna_objective(trial, X, y_true, C, file_name, root):
    D = trial.suggest_float("D", 1e-5, 0.3, step=0.001)
    dt = trial.suggest_float("dt", 1e-5, 0.1, step=0.001)

    stats = []
    times, times_graph, times_init = [], [], []
    runs = 3

    for _ in range(runs):
        y_rd, t_graph, t_init, t_solver = run_imex_reaction_diffusion(
            X, k=25, D=D, dt=dt, K=70, C=C, gpus=[0, 1, 2, 3]
        )
        times.append(t_solver + t_graph + t_init)
        times_graph.append(t_graph)
        times_init.append(t_init)
        stats.append(
            (
                adjusted_rand_score(y_true, y_rd),
                normalized_mutual_info_score(y_true, y_rd),
            )
        )

    aris = [s[0] for s in stats]
    nmis = [s[1] for s in stats]

    ari_mean, ari_std = np.mean(aris), np.std(aris)
    nmi_mean, nmi_std = np.mean(nmis), np.std(nmis)
    rd_mean, rd_std = np.mean(times), np.std(times)
    graph_mean, graph_std = np.mean(times_graph), np.std(times_graph)
    init_mean, init_std = np.mean(times_init), np.std(times_init)

    print(
        f"[Trial] D={D:.3f} | dt={dt:.3f} | "
        f"ARI={ari_mean:.2f}+-{ari_std:.2f} | "
        f"NMI={nmi_mean:.2f}+-{nmi_std:.2f} | "
        f"Time={rd_mean:.1f}ms"
    )

    save_run_to_xml(
        root,
        D,
        dt,
        graph_mean,
        graph_std,
        init_mean,
        init_std,
        rd_mean,
        rd_std,
        ari_mean,
        ari_std,
        nmi_mean,
        nmi_std,
    )
    return ari_mean


def compute_metrics_summary(name, stats, times):
    aris = [s[0] for s in stats[name]]
    nmis = [s[1] for s in stats[name]]
    tms = times[name]
    return (
        np.mean(aris),
        np.std(aris),
        np.mean(nmis),
        np.std(nmis),
        np.mean(tms),
        np.std(tms),
    )


def run_optuna_tuning(dataset_name, X, y_true, C):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n--- Starting dataset: {dataset_name} [{timestamp}]")

    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Started dataset {dataset_name}\n")

    os.makedirs("results_rd", exist_ok=True)
    file_name = f"results_rd/{timestamp.replace(':', '-')}_{dataset_name}.xml"
    root = ET.Element("experiment")

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: optuna_objective(trial, X, y_true, C, file_name, root),
        n_trials=10,
    )

    tree = ET.ElementTree(root)
    tree.write(file_name, encoding="utf-8", xml_declaration=True)

    print(f"Optimal params ({dataset_name}): {study.best_params}")
    print(f"Best ARI score: {study.best_value:.4f}")

    finished_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{finished_time}] Finished {dataset_name}\n")
        f.write(f"Best Params: {study.best_params} | Score: {study.best_value}\n")


def run_emnist_benchmark():
    print("\n--- EMNIST Benchmark ---")
    emnist_train = torchvision_datasets.EMNIST(root="./data", split="byclass", train=True, download=True)
    emnist_test = torchvision_datasets.EMNIST(root="./data", split="byclass", train=False, download=True)

    X_all = torch.cat([emnist_train.data, emnist_test.data], dim=0)
    y_all = torch.cat([emnist_train.targets, emnist_test.targets], dim=0)

    X_raw = X_all.view(-1, 28 * 28).numpy().astype("float32") / 255.0
    y_true = y_all.numpy().astype("int64")
    C = len(np.unique(y_true))

    print("Compressing EMNIST features using TruncatedSVD (n_components=100)...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_dense = svd.fit_transform(X_raw).astype("float32")

    runs = 5
    stats = {"rd": []}
    times = {"rd": []}

    for _ in range(runs):
        t0 = time.time()
        y_rd, _, _, _ = run_imex_reaction_diffusion(
            csr_matrix(X_dense), k=25, D=0.005, dt=0.1, K=70, C=C, gpus=[0, 1, 2, 3]
        )
        times["rd"].append((time.time() - t0) * 1000)
        stats["rd"].append(
            (
                adjusted_rand_score(y_true, y_rd),
                normalized_mutual_info_score(y_true, y_rd),
            )
        )

    m = compute_metrics_summary("rd", stats, times)
    print(
        f"[EMNIST Results] ARI = {m[0]:.2f}+-{m[1]:.2f} | "
        f"NMI = {m[2]:.2f}+-{m[3]:.2f} | "
        f"Time = {m[4]:.1f}+-{m[5]:.1f} ms"
    )


def process_dataset_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"[Error] Folder '{folder_path}' not found!")
        return

    dataset_files = [
        f for f in os.listdir(folder_path) if f.endswith(".npy") and "_labels" not in f
    ]

    if not dataset_files:
        print(f"[Warning] No .npy files found in '{folder_path}'.")
        return

    print(f"Found {len(dataset_files)} datasets in '{folder_path}'.")

    for file_name in sorted(dataset_files):
        base_name = file_name[:-4]
        x_path = os.path.join(folder_path, file_name)
        y_path = os.path.join(folder_path, f"{base_name}_labels.npy")

        X = np.load(x_path)

        if os.path.exists(y_path):
            y_true = np.load(y_path)
            n_clusters = len(np.unique(y_true))
            run_optuna_tuning(base_name, csr_matrix(X), y_true, n_clusters)
        else:
            print(f"[Warning] Labels missing for {base_name}. Skipping...")

        del X


def main():
    try:
        process_dataset_folder("synt_datasets")
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Fatal Error] {e}")
        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] Exception: {e}\n")
            f.write(traceback.format_exc())


if __name__ == "__main__":
    main()