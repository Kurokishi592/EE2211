import numpy as np
import matplotlib.pyplot as plt

class _RegNode:
    def __init__(self, depth, indices):
        self.depth = depth
        self.indices = indices
        self.threshold = None
        self.left = None
        self.right = None
        self.prediction = None

def _mse(y):
    if y.size == 0:
        return 0.0
    mu = np.mean(y)
    return float(np.mean((y - mu) ** 2))

def _best_threshold(X, y):
    # Candidates: midpoints between sorted unique feature values
    uniq = np.unique(X)
    if uniq.size <= 1:
        return None, None  # no split possible
    cands = (uniq[:-1] + uniq[1:]) / 2.0
    best_thr = None
    best_mse = np.inf
    base_mse = _mse(y)
    for thr in cands:
        left_mask = X <= thr
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue
        y_left = y[left_mask]
        y_right = y[right_mask]
        mse_left = _mse(y_left)
        mse_right = _mse(y_right)
        weighted = (mse_left * y_left.size + mse_right * y_right.size) / y.size
        if weighted < best_mse:
            best_mse = weighted
            best_thr = thr
    gain = base_mse - best_mse if best_thr is not None else 0.0
    return best_thr, gain

def auto_tree_regressor(X, y, max_depth, min_gain=1e-9, X_test=None, plot=True):
    """Automatic threshold decision tree regressor (1D feature).

    At each node selects the threshold that minimizes weighted MSE. Stops when
    max_depth reached, no valid threshold, or gain < min_gain.

    Prints MSE at each depth (weighted) and plots data with piecewise mean and vertical thresholds.
    """
    X = np.asarray(X).ravel()
    y = np.asarray(y).ravel()
    assert X.ndim == 1, "Only 1D X supported in auto_tree_regressor"

    n = X.size
    root = _RegNode(depth=0, indices=np.arange(n))
    nodes_by_depth = {0: [root]}

    for depth in range(max_depth):
        current_nodes = nodes_by_depth.get(depth, [])
        if not current_nodes:
            break
        next_nodes = []
        for i_node, node in enumerate(current_nodes):
            X_node = X[node.indices]
            y_node = y[node.indices]
            node.prediction = float(np.mean(y_node))
            if depth == max_depth:
                continue
            thr, gain = _best_threshold(X_node, y_node)
            if thr is None or gain < min_gain:
                continue
            left_mask = X_node <= thr
            right_mask = ~left_mask
            left_indices = node.indices[left_mask]
            right_indices = node.indices[right_mask]
            node.threshold = thr
            node.left = _RegNode(depth=depth + 1, indices=left_indices)
            node.right = _RegNode(depth=depth + 1, indices=right_indices)
            # per-child MSE prints
            y_left = y[left_indices]
            y_right = y[right_indices]
            mse_left = _mse(y_left)
            mse_right = _mse(y_right)
            print(f"[AutoTreeRegressor] Depth {depth} Node {i_node} split @ {thr:.4f} -> left MSE: {mse_left:.4f}, right MSE: {mse_right:.4f}")
            next_nodes.extend([node.left, node.right])
        if next_nodes:
            nodes_by_depth[depth + 1] = next_nodes

    # finalize predictions for leaves
    for depth, nodes in nodes_by_depth.items():
        for node in nodes:
            if node.prediction is None:
                y_node = y[node.indices]
                node.prediction = float(np.mean(y_node))

    depth_mse = {}
    for depth, nodes in nodes_by_depth.items():
        total = 0.0
        count = 0
        for node in nodes:
            y_node = y[node.indices]
            mse_node = _mse(y_node)
            total += mse_node * y_node.size
            count += y_node.size
        depth_mse[depth] = total / count if count else 0.0
        print(f"[AutoTreeRegressor] Depth {depth} weighted MSE: {depth_mse[depth]:.4f}")

    train_pred = np.zeros_like(y, dtype=float)
    for depth, nodes in nodes_by_depth.items():
        for node in nodes:
            train_pred[node.indices] = node.prediction

    test_pred = None
    if X_test is not None:
        X_test = np.asarray(X_test).ravel()
        test_pred = np.zeros_like(X_test, dtype=float)
        for i, xval in enumerate(X_test):
            current = root
            while current.threshold is not None and current.left is not None:
                if xval <= current.threshold:
                    current = current.left
                else:
                    current = current.right
            test_pred[i] = current.prediction

    thresholds_used = [node.threshold for depth in sorted(nodes_by_depth) for node in nodes_by_depth[depth] if node.threshold is not None]

    if plot:
        sorted_pairs = sorted(zip(X, y), key=lambda t: t[0])
        Xs = np.array([p[0] for p in sorted_pairs])
        Ys = np.array([p[1] for p in sorted_pairs])
        plt.figure(figsize=(8, 4))
        plt.scatter(Xs, Ys, color='steelblue', s=30, label='Data')
        grid = np.linspace(Xs.min(), Xs.max(), 300)
        grid_pred = np.zeros_like(grid)
        for gi, gx in enumerate(grid):
            current = root
            while current.threshold is not None and current.left is not None:
                if gx <= current.threshold:
                    current = current.left
                else:
                    current = current.right
            grid_pred[gi] = current.prediction
        plt.plot(grid, grid_pred, color='red', lw=2, label='Piecewise mean')
        for thr in thresholds_used:
            plt.axvline(thr, color='purple', linestyle='--', alpha=0.7)
        plt.title('Auto Tree Regressor')
        plt.xlabel('X')
        plt.ylabel('y / prediction')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        'thresholds': thresholds_used,
        'depth_mse': depth_mse,
        'train_pred': train_pred,
        'test_pred': test_pred,
    }
