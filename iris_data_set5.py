# 📦 Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 🌸 Load Iris Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print("Original X shape:", X.shape)

# ✂️ Split Dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# 🤖 Train Models
dt_model = DecisionTreeClassifier().fit(X_train, y_train)
knn_model = KNeighborsClassifier().fit(X_train, y_train)
svm_model = SVC().fit(X_train, y_train)

# 📊 Evaluate Accuracy
dt_acc = dt_model.score(X_test, y_test)
knn_acc = knn_model.score(X_test, y_test)
svm_acc = svm_model.score(X_test, y_test)

print("\n🌳 Decision Tree Accuracy:", dt_acc)
print("📍 KNN Accuracy:", knn_acc)
print("🧠 SVM Accuracy:", svm_acc)

# 🧾 Classification Reports
models = {'Decision Tree': dt_model, 'KNN': knn_model, 'SVM': svm_model}
for name, model in models.items():
    print(f"\n===== 📊 {name} Evaluation =====")
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# 🧠 Predict on New Input
print("\n📥 Enter values to predict the Iris species:")
sl = float(input("Sepal Length (cm): "))
sw = float(input("Sepal Width (cm): "))
pl = float(input("Petal Length (cm): "))
pw = float(input("Petal Width (cm): "))
user_input = np.array([[sl, sw, pl, pw]])

print("\n🔮 Predictions for given input:")
print("🌳 Decision Tree Prediction:", target_names[dt_model.predict(user_input)[0]])
print("📍 KNN Prediction:", target_names[knn_model.predict(user_input)[0]])
print("🧠 SVM Prediction:", target_names[svm_model.predict(user_input)[0]])

# 📈 Visualization (2 Features: Petal Length vs Width)
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 2], X[y == i, 3], label=target_name, color=colors[i])
    
plt.scatter(pl, pw, color='black', label='Your Input', marker='X', s=100)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Dataset - Petal Length vs Width")
plt.legend()
plt.grid(True)
plt.show()

# 🏆 Best Model Conclusion
accs = {'Decision Tree': dt_acc, 'KNN': knn_acc, 'SVM': svm_acc}
best_model = max(accs, key=accs.get)

print(f"\n✅ Best Model: {best_model} (based on accuracy)")
print(f"Reason: Achieved highest accuracy on test data ({accs[best_model] * 100:.2f}%)")
