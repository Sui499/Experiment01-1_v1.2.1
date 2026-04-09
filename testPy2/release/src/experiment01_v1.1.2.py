import os
import sys
import subprocess


# ==========================================
# 0. 全自动环境依赖检查与静默安装
# ==========================================
def auto_setup_environment():
    """全自动检查并静默安装所有必需的第三方库"""
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn']
    python_exe = sys.executable

    print("=" * 60)
    print("🚀 启动全自动环境自检...")

    # 1. 优先尝试通过同一目录下的 requirements.txt 安装
    if os.path.exists("requirements.txt"):
        try:
            print("📦 正在根据 requirements.txt 同步环境，请稍候...")
            subprocess.check_call([python_exe, "-m", "pip", "install", "-r", "requirements.txt"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception:
            pass

            # 2. 逐个检查核心库，缺失则自动静默安装
    for package in required_packages:
        # sklearn 在 pip 中叫 scikit-learn，在代码中 import 叫 sklearn
        check_name = 'sklearn' if package == 'scikit-learn' else package
        try:
            __import__(check_name)
        except ImportError:
            print(f"🔍 正在后台自动安装缺失库: {package}...")
            try:
                subprocess.check_call([python_exe, "-m", "pip", "install", package],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            except Exception as e:
                print(f"❌ 自动安装 {package} 失败，请手动执行: pip install {package}")
                sys.exit(1)

    print("✅ 环境准备就绪，正在启动实验程序...")
    print("=" * 60 + "\n")


# 在导入大型库之前执行自检
auto_setup_environment()











import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score

# ==========================================
# 1. 环境设置与数据预处理
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 【新增】创建输出目录
os.makedirs("../out", exist_ok=True)

# 读取数据
df = pd.read_csv("../data/StudentPerformanceFactors.csv")

# 【核心修复】：必须加 .iloc[0] 才能提取出具体的数值，否则会报错 _iLocIndexer
df = df.fillna(df.mode().iloc[0])

# 【非数值转数值】：使用 LabelEncoder 自动化转换所有类别特征
le = LabelEncoder()
string_cols = df.select_dtypes(include=['object', 'string']).columns
for col in string_cols:
    df[col] = le.fit_transform(df[col])

# ==========================================
# 2. 相关性分析（保留原始输出与图表）
# ==========================================
corrs = df.corr()['Exam_Score'].drop('Exam_Score')
strong_corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
top_10_features = strong_corrs.head(10)

print("--- 实验数据显示出：与学习成绩强相关的项 ---")
print(top_10_features)

# 可视化相关性
plt.figure(figsize=(10, 6))
colors = ['#ff7675' if x < 0 else '#55efc4' for x in top_10_features.values]
top_10_features.plot(kind='barh', color=colors)
plt.axvline(x=0, color='black', linestyle='-')
plt.title('Exam_Score 强相关因子（红色负影响，绿色正影响）')
plt.tight_layout()
plt.savefig("../out/correlation_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 3. 降维与特征工程
# ==========================================
X = df[top_10_features.index.tolist()]
y = df['Exam_Score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.9)
X_pca = pca.fit_transform(X_scaled)

# ==========================================
# 4. 回归算法实验：方案 B (10折验证) + 原始输出
# ==========================================
# --- 方案 B: 交叉验证评估 ---
lr_cv_model = LinearRegression()
cv_r2 = cross_val_score(lr_cv_model, X_pca, y, cv=10, scoring='r2')

# --- 保留原始单次运行逻辑 ---
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)
lr = LinearRegression()
lr.fit(X_train, y_train)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

print(f"\n{'='*20} 方案 B: 10折交叉验证 (回归) {'='*20}")
print(f"[回归评估] 线性回归 10-Fold 平均 R²: {cv_r2.mean():.4f} (±{cv_r2.std():.4f})")

print(f"\n--- 原始回归指标输出 ---")
print(f"[回归评价指标] LSM R²: {r2_score(y_test, lr.predict(X_test)):.4f}")
print(f"[回归评价指标] Ridge MSE: {mean_squared_error(y_test, ridge.predict(X_test)):.4f}")
print(f"[回归评价指标] MAE: {mean_absolute_error(y_test, lr.predict(X_test)):.4f}")

# 回归结果可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
y_pred_lr = lr.predict(X_test)
plt.scatter(y_test, y_pred_lr, alpha=0.6, color='#0984e3')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('线性回归：实际值 vs 预测值')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
y_pred_ridge = ridge.predict(X_test)
plt.scatter(y_test, y_pred_ridge, alpha=0.6, color='#6c5ce7')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('Ridge回归：实际值 vs 预测值')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../out/regression_results.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 5. 分类算法实验：方案 B (10折验证) + 原始输出
# ==========================================
y_class = (y > y.median()).astype(int)

# --- 方案 B: 交叉验证评估 ---
skf = StratifiedKFold(n_splits=10, shuffle=True)
models_to_test = {
    "KNN (KD-Tree)": KNeighborsClassifier(n_neighbors=5),
    "决策树 (CART)": DecisionTreeClassifier(max_depth=5),
    "朴素贝叶斯": GaussianNB()
}

print(f"\n{'='*20} 方案 B: 10折交叉验证 (分类) {'='*20}")
cv_results = {}
for name, m in models_to_test.items():
    scores = cross_val_score(m, X_pca, y_class, cv=skf, scoring='accuracy')
    cv_results[name] = scores
    print(f"[分类评估] {name:15} 10-Fold 平均准确率: {scores.mean():.4f} (±{scores.std():.4f})")

# 分类交叉验证结果可视化
plt.figure(figsize=(10, 6))
model_names = list(cv_results.keys())
means = [cv_results[m].mean() for m in model_names]
stds = [cv_results[m].std() for m in model_names]
colors_bar = ['#00b894', '#fdcb6e', '#e17055']
plt.bar(model_names, means, yerr=stds, capsize=5, color=colors_bar, alpha=0.7, edgecolor='black')
plt.ylabel('准确率')
plt.title('10折交叉验证 - 分类模型对比')
plt.ylim([0.75, 0.95])
for i, (m, s) in enumerate(zip(means, stds)):
    plt.text(i, m + s + 0.01, f'{m:.4f}', ha='center', va='bottom', fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("../out/classification_cv_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# --- 保留原始手写算法与单次运行输出 ---
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_pca, y_class, test_size=0.2)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def mini_batch_gd(X, y_true, lr_rate=0.1, epochs=50):
    w = np.zeros(X.shape[1])  # 【修复】改为 X.shape[1]，获取特征数量
    b = 0
    y_vals = y_true.values
    for _ in range(epochs):
        idx = np.random.permutation(len(y_vals))
        Xs, ys = X[idx], y_vals[idx]
        for i in range(0, len(ys), 32):
            xi, yi = Xs[i:i+32], ys[i:i+32]
            y_hat = sigmoid(np.dot(xi, w) + b)
            w -= lr_rate * (1/len(yi)) * np.dot(xi.T, (y_hat - yi))
            b -= lr_rate * (1/len(yi)) * np.sum(y_hat - yi)
    return w, b

w, b = mini_batch_gd(Xc_train, yc_train)
mbgd_preds = (sigmoid(np.dot(Xc_test, w) + b) > 0.5).astype(int)

# B/C/D 算法单次训练
knn = KNeighborsClassifier(n_neighbors=5).fit(Xc_train, yc_train)
dt = DecisionTreeClassifier(max_depth=5).fit(Xc_train, yc_train)
gnb = GaussianNB().fit(Xc_train, yc_train)

print(f"\n--- 原始分类指标输出 ---")
mbgd_acc = accuracy_score(yc_test, mbgd_preds)
knn_acc = accuracy_score(yc_test, knn.predict(Xc_test))
dt_acc = accuracy_score(yc_test, dt.predict(Xc_test))
gnb_acc = accuracy_score(yc_test, gnb.predict(Xc_test))

print(f"[分类准确率] MBGD逻辑回归: {mbgd_acc:.4f}")
print(f"[分类准确率] KNN (KD-Tree): {knn_acc:.4f}")
print(f"[分类准确率] 决策树 (CART): {dt_acc:.4f}")
print(f"[分类准确率] 朴素贝叶斯: {gnb_acc:.4f}")

# 单次测试结果可视化
plt.figure(figsize=(10, 6))
test_models = ['MBGD逻辑回归', 'KNN (KD-Tree)', '决策树 (CART)', '朴素贝叶斯']
test_accuracies = [mbgd_acc, knn_acc, dt_acc, gnb_acc]
colors_test = ['#a29bfe', '#fd79a8', '#fdcb6e', '#00b894']
plt.bar(test_models, test_accuracies, color=colors_test, alpha=0.7, edgecolor='black')
plt.ylabel('准确率')
plt.title('单次测试集 - 分类模型准确率对比')
plt.ylim([0.75, 0.95])
for i, acc in enumerate(test_accuracies):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom', fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("../out/classification_test_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ 所有图表已保存到 ../out 目录")