# 📦 Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 📁 Create Folder for Saving Outputs
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# 🌸 Load Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# ✂️ Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔄 Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🤖 Train Models
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
svm_model = SVC(kernel='rbf', C=1)

dt_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# 📊 Accuracy
dt_acc = dt_model.score(X_test, y_test)
knn_acc = knn_model.score(X_test, y_test)
svm_acc = svm_model.score(X_test, y_test)

# 💾 Save Accuracy to File
with open(f"{output_dir}/accuracy.txt", "w") as f:
    f.write(f"Decision Tree Accuracy: {dt_acc}\n")
    f.write(f"KNN Accuracy: {knn_acc}\n")
    f.write(f"SVM Accuracy: {svm_acc}\n")

# 📊 Confusion Matrix (Saved as Image)
models = {'Decision Tree': dt_model, 'KNN': knn_model, 'SVM': svm_model}

for name, model in models.items():
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{output_dir}/{name}_confusion.png")
    plt.close()

    # 💾 Save Classification Report
    report = classification_report(y_test, y_pred, target_names=target_names)
    with open(f"{output_dir}/{name}_report.txt", "w") as f:
        f.write(report)

# 🔮 Demo Input (Auto instead of manual input for GitHub)
demo_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # Setosa example
demo_scaled = scaler.transform(demo_input)

# 💾 Save Predictions
with open(f"{output_dir}/prediction.txt", "w") as f:
    f.write("Demo Input: [5.1, 3.5, 1.4, 0.2]\n\n")
    f.write(f"Decision Tree: {target_names[dt_model.predict(demo_scaled)[0]]}\n")
    f.write(f"KNN: {target_names[knn_model.predict(demo_scaled)[0]]}\n")
    f.write(f"SVM: {target_names[svm_model.predict(demo_scaled)[0]]}\n")

# 📈 Visualization
plt.figure(figsize=(8, 5))
colors = ['red', 'green', 'blue']

for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 2], X[y == i, 3],
                label=target_name, color=colors[i])

# Mark Demo Input
plt.scatter(demo_input[0][2], demo_input[0][3],
            color='black', marker='X', s=100, label='Demo Input')

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Dataset Visualization")
plt.legend()
plt.grid(True)

# 💾 Save Graph
plt.savefig(f"{output_dir}/graph.png")
plt.close()

print("✅ All outputs saved in 'images/' folder!")