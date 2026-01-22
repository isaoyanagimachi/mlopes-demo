import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import joblib
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
clf = RandomForestClassifier(max_depth=7, random_state=0)
clf.fit(X, y)
joblib.dump(clf, "model.pkl")

# =========================
# 2. Connect to Workspace
# =========================
from azureml.core import Workspace
ws = Workspace.from_config()
print("Workspace loaded:", ws.name)

# =========================
# 3. Register model
# =========================
from azureml.core import Model
model = Model.register(
    workspace=ws,
    model_path="model.pkl",
    model_name="simple_iris_model"
)
print("Registered model:", model.name, model.version)
