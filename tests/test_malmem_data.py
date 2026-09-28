import pandas as pd

from src.data.malmem_data import load_malmem_data


def test_malmem_has_55_model_features():
    features, target, metadata = load_malmem_data()

    assert features.shape[1] == 55
    assert target.name == "Class"
    assert "Category" in metadata.columns
    assert "Filename" in metadata.columns


def test_malmem_features_are_numeric():
    features, _, _ = load_malmem_data()

    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes)
def test_malmem_split_has_no_feature_vector_leakage():
    from src.data.malmem_data import split_malmem_data

    X_train, X_test, y_train, y_test = split_malmem_data(
        test_size=0.2,
        random_state=42,
    )

    train_vectors = {
        tuple(row)
        for row in X_train.to_numpy()
    }

    test_vectors = {
        tuple(row)
        for row in X_test.to_numpy()
    }

    assert train_vectors.isdisjoint(test_vectors)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == 58596
def test_malmem_split_is_reproducible():
    from src.data.malmem_data import split_malmem_data

    first = split_malmem_data(
        test_size=0.2,
        random_state=42,
    )

    second = split_malmem_data(
        test_size=0.2,
        random_state=42,
    )

    for first_part, second_part in zip(first, second):
        assert first_part.equals(second_part)
def test_malmem_split_contains_both_classes():
    from src.data.malmem_data import split_malmem_data

    _, _, y_train, y_test = split_malmem_data(
        test_size=0.2,
        random_state=42,
    )

    assert set(y_train.unique()) == {"Benign", "Malware"}
    assert set(y_test.unique()) == {"Benign", "Malware"}
