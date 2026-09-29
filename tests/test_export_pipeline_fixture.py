"""Tests for the forest export behind the app's "inside the forest" view."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.export_pipeline_fixture import build_forest, sample_tree_paths


def _tiny_tree():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 100, size=(200, 3))
    y = X[:, 0] * 0.5 + (X[:, 1] > 50) * 20
    return DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y), X


def test_paths_run_root_to_leaf_through_parent_child_links():
    tree, X = _tiny_tree()
    t = tree.tree_
    for path in sample_tree_paths(tree, X[:25]):
        assert path[0] == 0
        assert t.children_left[path[-1]] == -1  # ends at a leaf
        for parent, child in zip(path, path[1:]):
            assert child in (t.children_left[parent], t.children_right[parent])


def test_path_directions_match_the_split_rule():
    tree, X = _tiny_tree()
    t = tree.tree_
    for row, path in zip(X[:25], sample_tree_paths(tree, X[:25])):
        for parent, child in zip(path, path[1:]):
            went_left = child == t.children_left[parent]
            assert went_left == (row[t.feature[parent]] <= t.threshold[parent])


def test_leaf_value_is_the_trees_prediction():
    tree, X = _tiny_tree()
    for row, path in zip(X[:25], sample_tree_paths(tree, X[:25])):
        assert tree.tree_.value[path[-1]][0][0] == tree.predict(row.reshape(1, -1))[0]


def test_build_forest_records_split_values_and_only_visited_nodes():
    tree, X = _tiny_tree()
    columns = ["a", "b", "c"]
    df = pd.DataFrame(X[:10], columns=columns)
    df["unit_number"] = 7
    df["tree_predictions"] = [[1, 2, 3]] * 10
    df["sample_tree_path"] = sample_tree_paths(tree, X[:10])

    forest = build_forest(df, tree, columns)

    visited = {n for path in df["sample_tree_path"] for n in path}
    assert set(map(int, forest["sample_tree"]["nodes"])) == visited
    first_path = forest["sample_tree"]["paths"][0]
    assert len(forest["sample_tree"]["split_values"][0]) == len(first_path) - 1
    root = forest["sample_tree"]["nodes"]["0"]
    assert forest["sample_tree"]["split_values"][0][0] == float(f"{X[0, root['feature']]:.6g}")
    assert "value" in forest["sample_tree"]["nodes"][str(first_path[-1])]
