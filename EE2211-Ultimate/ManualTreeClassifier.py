import numpy as np
import matplotlib.pyplot as plt

class _ClsNode:
    def __init__(self, depth, indices):
        self.depth = depth
        self.indices = indices
        self.threshold = None
        self.left = None
        self.right = None
        self.prediction = None  # majority class

def _gini(y):
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    probs = counts / y.size
    return float(1.0 - np.sum(probs ** 2))

def _entropy(y):
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    probs = counts / y.size
    # avoid log(0) by masking
    mask = probs > 0
    return float(-np.sum(probs[mask] * np.log2(probs[mask])))

def _misclassification(y):
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p_max = counts.max() / y.size
    return float(1.0 - p_max)

def _impurity(y, criterion: str):
    c = criterion.lower()
    if c == 'gini':
        return _gini(y)
    if c == 'entropy':
        return _entropy(y)
    if c in ('misclassification', 'misclass', 'error'):
        return _misclassification(y)
    raise ValueError(f"Unknown criterion '{criterion}'. Use 'gini', 'entropy', or 'misclassification'.")

def _get_node_threshold(decision_threshold, depth, node_index):
    """Return threshold for specific node given nested list format.

    Formats:
      - Scalar: same threshold everywhere.
      - Nested list: [root_thr, [d1_left, d1_right], [d2_n0, d2_n1, d2_n2, d2_n3], ...]
        Missing depths/elements => no split for that node.
    """
    if np.isscalar(decision_threshold):
        try:
            return float(decision_threshold)
        except Exception:
            return None
    try:
        dt = list(decision_threshold)
    except Exception:
        return None
    if depth >= len(dt):
        return None
    if depth == 0:
        val = dt[0]
        if np.isscalar(val):
            try:
                return float(val)
            except Exception:
                return None
        try:
            return float(val[0])
        except Exception:
            return None
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

def manual_tree_classifier(X, y, max_depth, decision_threshold, X_test=None, plot=True, criterion: str = 'gini'):
    """Manual decision tree classifier (1D feature) with explicit per-node thresholds.

    decision_threshold formats:
        - Scalar: same threshold for every node.
        - Nested list: [root_thr, [d1_left, d1_right], [d2_n0, d2_n1, d2_n2, d2_n3], ...]

    Splitting rule: left = X <= t, right = X > t.
    Missing threshold => node becomes leaf.

    Prints per-node impurity and weighted per-depth impurity using the chosen criterion.
    """
    X = np.asarray(X).ravel()
    y = np.asarray(y).ravel()
    max_depth = int(max_depth)  # interpret as leaf depth, root depth = 0
    assert X.ndim == 1, "Only 1D X supported in manual_tree_classifier"

    root = _ClsNode(depth=0, indices=np.arange(X.size))
    nodes_by_depth = {0: [root]}

    for depth in range(max_depth):  # will create children up to depth == max_depth
        current_nodes = nodes_by_depth.get(depth, [])
        if not current_nodes:
            break
        next_nodes = []
        for i_node, node in enumerate(current_nodes):
            X_node = X[node.indices]
            y_node = y[node.indices]
            # majority class prediction
            vals, counts = np.unique(y_node, return_counts=True)
            node.prediction = int(vals[np.argmax(counts)])
            # per-node impurity print
            imp = _impurity(y_node, criterion)
            print(f"[ManualTreeClassifier] Depth {depth} Node {i_node} size {y_node.size} impurity({criterion}): {imp:.4f} pred: {node.prediction}")
            if (depth + 1) > max_depth or X_node.size <= 1:
                continue
            thr_d = _get_node_threshold(decision_threshold, depth, i_node)
            if thr_d is None:
                continue  # no threshold provided for this depth
            left_mask = X_node <= thr_d
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            node.threshold = thr_d
            left_indices = node.indices[left_mask]
            right_indices = node.indices[right_mask]
            node.left = _ClsNode(depth=depth + 1, indices=left_indices)
            node.right = _ClsNode(depth=depth + 1, indices=right_indices)
            next_nodes.extend([node.left, node.right])
        if next_nodes:
            nodes_by_depth[depth + 1] = next_nodes

    # finalize predictions
    for depth, nodes in nodes_by_depth.items():
        for node in nodes:
            if node.prediction is None:
                y_node = y[node.indices]
                vals, counts = np.unique(y_node, return_counts=True)
                node.prediction = int(vals[np.argmax(counts)])

    # Depth impurity
    depth_imp = {}
    for depth, nodes in nodes_by_depth.items():
        total = 0.0
        count = 0
        for node in nodes:
            y_node = y[node.indices]
            imp = _impurity(y_node, criterion)
            total += imp * y_node.size
            count += y_node.size
        depth_imp[depth] = total / count if count else 0.0
        print(f"[ManualTreeClassifier] Depth {depth} weighted {criterion}: {depth_imp[depth]:.4f}")

    # Train predictions
    train_pred = np.zeros_like(y)
    for depth, nodes in nodes_by_depth.items():
        for node in nodes:
            train_pred[node.indices] = node.prediction

    test_pred = None
    if X_test is not None:
        X_test = np.asarray(X_test).ravel()
        test_pred = np.zeros_like(X_test)
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
        plt.scatter(Xs, Ys, c=Ys, cmap='viridis', s=40, label='Data')
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
        plt.plot(grid, grid_pred, color='red', lw=2, label='Predicted class')
        for thr in thresholds_used:
            plt.axvline(thr, color='green', linestyle='--', alpha=0.7)
        plt.title('Manual Tree Classifier')
        plt.xlabel('X')
        plt.ylabel('Class')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        'thresholds': thresholds_used,
        'depth_gini': depth_imp,
        'train_pred': train_pred,
        'test_pred': test_pred,
    }
