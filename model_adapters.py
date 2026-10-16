import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class XGBoostLabelAdapter(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from xgboost import XGBClassifier

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(np.asarray(y))

        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            eval_metric=self.eval_metric,
            objective="binary:logistic",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y_encoded)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        predictions = np.asarray(self.model_.predict(X)).astype(int).ravel()
        return self.label_encoder_.inverse_transform(predictions)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
