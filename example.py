from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# utils.py and separation_mvp.py are in the repo
from utils import classification_report
from separation_mvp import SeparatedClassifier

url = "https://raw.githubusercontent.com/omarfsosa/datasets/master/fairness_synthetic_data.csv"
df = pd.read_csv(url)

X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    df.drop(columns="y"),
    df["y"],
    df["A"],  # (2)
    test_size=.6,  # (3)
    random_state=42,
)

clf = LogisticRegression(solver="lbfgs")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, A_test))  # (4)

R_train = clf.predict_proba(X_train)[:, 1]
R_test = clf.predict_proba(X_test)[:, 1]
goal_tpr, goal_fpr = 0.83591123066577, 0.2639968121139669


fair_clf = SeparatedClassifier(y_train, R_train, A_train)
fair_clf.fit(goal_fpr, goal_tpr)

for k, v in fair_clf.randomized_thresholds.items():
    print(f"Group {k}: t0={v[0]:.2f}, t1={v[1]:.2f}, p={v[2]:.2f}")

y_pred_fair = fair_clf.fair_predict(R_test, A_test)
print(classification_report(y_test, y_pred_fair, A_test))
