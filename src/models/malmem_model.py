from sklearn.ensemble import RandomForestClassifier


def train_malmem_model(
    X_train,
    y_train,
    random_state=42,
):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model
