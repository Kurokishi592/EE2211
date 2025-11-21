import numpy as np
import matplotlib.pyplot as plt

class _RegNode:
    def __init__(self, depth, indices):
        self.depth = depth
        self.indices = indices  # indices of samples in this node
        self.threshold = None
        self.left = None
        self.right = None
        self.prediction = None  # mean y

def _mse(y):
    if y.size == 0:
        return 0.0
    mu = np.mean(y)
    return float(np.mean((y - mu) ** 2))

def _get_node_threshold(decision_threshold, depth, node_index):
    """Return threshold for a specific node given nested per-depth specification.

    Supported input formats:
      1. Scalar (int/float): same threshold for all nodes.
      2. List/tuple in nested per-depth form:
         [root_threshold, [d1_left, d1_right], [d2_n0, d2_n1, d2_n2, d2_n3], ...]

    If a depth list is missing or too short, returns None (no split for that node).
    """
    # Scalar threshold broadcast to all nodes
    if np.isscalar(decision_threshold):
        try:
            return float(decision_threshold)
        except Exception:
            return None
    # Must be an indexable sequence
    try:
        dt = list(decision_threshold)
    except Exception:
        return None
    if depth >= len(dt):
        return None
    # Depth 0 root threshold element expected to be scalar (not a list)
    if depth == 0:
        val = dt[0]
        if np.isscalar(val):
            try:
                return float(val)
            except Exception:
                return None
        # If user mistakenly provided list for root we take first element
        try:
            return float(val[0])
        except Exception:
            return None
    # Depth > 0 expects a list/sequence of thresholds of length >= node_index+1
    depth_list = dt[depth]
    try:
        depth_list = list(depth_list)
    except Exception:
        return None
    if node_index >= len(depth_list):
        return None
    try:
        return float(depth_list[node_index])
    except Exception:
        return None


def manual_tree_regressor(X, y, max_depth, decision_threshold, X_test=None, plot=True):
    """Manual decision tree regressor (1D feature) with explicit per-node thresholds.

    decision_threshold formats:
      - Scalar (int/float): same threshold at all nodes.
      - Nested list: [root_thr, [depth1_left, depth1_right], [depth2_n0, depth2_n1, depth2_n2, depth2_n3], ...]
        Each depth list supplies thresholds for nodes left-to-right at that depth.
        If a depth list or an element is missing, that node becomes a leaf.

    Example:
        decision_threshold = [3.0, [4.0, 5.2], [4.5, 4.8, 5.1, 5.6]]

    Splitting: left = X <= t, right = X > t.
    Stopping: reached max_depth, missing threshold, node too small, or split makes empty child.

    Returns:
        dict(thresholds, depth_mse, train_pred, test_pred)
    """
    X = np.asarray(X).ravel()
    y = np.asarray(y).ravel()
    max_depth = int(max_depth)  # interpret as leaf depth, root depth = 0
    assert X.ndim == 1, "Only 1D X supported in manual_tree_regressor"

    n = X.size
    indices = np.arange(n)
    root = _RegNode(depth=0, indices=indices)
    nodes_by_depth = {0: [root]}

    # Build tree iteratively breadth-first
    for depth in range(max_depth):  # will create children up to depth == max_depth
        current_nodes = nodes_by_depth.get(depth, [])
        if not current_nodes:
            break
        next_depth_nodes = []
        for i_node, node in enumerate(current_nodes):
            X_node = X[node.indices]
            y_node = y[node.indices]
            node.prediction = float(np.mean(y_node))
            # Decide whether to split (avoid creating children deeper than max_depth)
            if (depth + 1) > max_depth or X_node.size <= 1:
                continue
            thr_d = _get_node_threshold(decision_threshold, depth, i_node)
            if thr_d is None:
                continue  # no threshold for this node
            left_mask = X_node <= thr_d
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue  # cannot split further
            node.threshold = thr_d
            left_indices = node.indices[left_mask]
            right_indices = node.indices[right_mask]
            node.left = _RegNode(depth=depth + 1, indices=left_indices)
            node.right = _RegNode(depth=depth + 1, indices=right_indices)
            # per-child MSE prints
            y_left = y[left_indices]
            y_right = y[right_indices]
            mse_left = _mse(y_left)
            mse_right = _mse(y_right)
            print(f"[ManualTreeRegressor] Depth {depth} Node {i_node} split @ {thr_d:.4f} -> left MSE: {mse_left:.4f}, right MSE: {mse_right:.4f}")
            next_depth_nodes.extend([node.left, node.right])
        if next_depth_nodes:
            nodes_by_depth[depth + 1] = next_depth_nodes

    # Assign predictions to leaves
    for depth, nodes in nodes_by_depth.items():
        for node in nodes:
            if node.prediction is None:
                y_node = y[node.indices]
                node.prediction = float(np.mean(y_node))

    # Compute depth MSE (weighted by samples in each node)
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
        print(f"[ManualTreeRegressor] Depth {depth} weighted MSE: {depth_mse[depth]:.4f}")

    # Prepare train predictions (piecewise constant)
    train_pred = np.zeros_like(y, dtype=float)
    for depth, nodes in nodes_by_depth.items():
        for node in nodes:
            train_pred[node.indices] = node.prediction

    # Test predictions (same piecewise rule)
    test_pred = None
    if X_test is not None:
        X_test = np.asarray(X_test).ravel()
        test_pred = np.zeros_like(X_test, dtype=float)
        for i, xval in enumerate(X_test):
            # Traverse using threshold until leaf
            current = root
            while current.threshold is not None and current.left is not None:
                if xval <= current.threshold:
                    current = current.left
                else:
                    current = current.right
            test_pred[i] = current.prediction

    thresholds_used = [node.threshold for depth in sorted(nodes_by_depth) for node in nodes_by_depth[depth] if node.threshold is not None]

    if plot:
        # Create intervals from thresholds
        sorted_pairs = sorted(zip(X, y), key=lambda t: t[0])
        Xs = np.array([p[0] for p in sorted_pairs])
        Ys = np.array([p[1] for p in sorted_pairs])
        plt.figure(figsize=(8, 4))
        plt.scatter(Xs, Ys, color='steelblue', s=30, label='Data')
        # For visualization create fine grid predictions
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
            plt.axvline(thr, color='green', linestyle='--', alpha=0.7)
        plt.title('Manual Tree Regressor')
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
