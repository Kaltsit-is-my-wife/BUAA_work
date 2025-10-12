import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def file2matrix(filename):
    with open(filename, 'r', encoding='utf-8') as fr:
        lines = [line.strip() for line in fr if line.strip()]
    m = len(lines)
    X = np.zeros((m, 3), dtype=float)
    y = np.zeros(m, dtype=int)
    for i, line in enumerate(lines):
        parts = line.split('\t')
        X[i, :] = list(map(float, parts[0:3]))
        y[i] = int(parts[-1])
    return X, y


def optimize_and_plot(train_file='data/knn_trainData.txt', test_file='data/knn_testData.txt'):
    X_train, y_train = file2matrix(train_file)
    X_test, y_test = file2matrix(test_file)

    # 管道 + 网格
    pipe = Pipeline(steps=[
        ('scaler', MinMaxScaler()),   # 仅占位，实际由 param_grid 覆盖
        ('knn', KNeighborsClassifier())
    ])

    param_grid = [
        {
            'scaler': [MinMaxScaler()],
            'knn__n_neighbors': list(range(1, 51)),
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['manhattan']
        },
        {
            'scaler': [StandardScaler()],
            'knn__n_neighbors': list(range(1, 51)),
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean']
        }
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        return_train_score=False,
        refit=True
    )
    grid.fit(X_train, y_train)

    print('Best CV params:', grid.best_params_)
    print(f'Best CV accuracy: {grid.best_score_:.4f}')

    # 最终测试集评估
    y_pred = grid.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f'Test accuracy (best model): {test_acc:.4f}')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification report:\n', classification_report(y_test, y_pred, digits=4))

    # 画“验证曲线”：对每个 K 取“该 K 下所有组合的最高 CV 均值”，并给出该 K 下所有组合的标准差范围
    results = grid.cv_results_
    ks = np.array(results['param_knn__n_neighbors'].data, dtype=int)
    means = np.array(results['mean_test_score'])
    stds = np.array(results['std_test_score'])

    unique_k = sorted(set(ks))
    best_mean_per_k = []
    std_per_k = []

    for k in unique_k:
        mask = (ks == k)
        best_mean_per_k.append(means[mask].max())
        # 用该 K 下所有组合的标准差的均值作参考带（也可用对应最大均值那条的 std）
        std_per_k.append(stds[mask].mean())

    # 另外，固定“最优缩放+最优度量”，比较 uniform vs distance 在测试集上的曲线（只画 1..30）
    best = grid.best_params_
    # 重新拟合 scaler（只用训练集），然后逐 K 评估在测试集上的 accuracy
    if isinstance(best['scaler'], MinMaxScaler):
        scaler = MinMaxScaler().fit(X_train)
        metric = best['knn__metric']  # 'manhattan'
    else:
        scaler = StandardScaler().fit(X_train)
        metric = best['knn__metric']  # 'euclidean'

    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    test_acc_uniform = []
    test_acc_distance = []
    k_range_show = list(range(1, 31))
    for k in k_range_show:
        for w, bucket in [('uniform', test_acc_uniform), ('distance', test_acc_distance)]:
            clf = KNeighborsClassifier(n_neighbors=k, weights=w, metric=metric)
            clf.fit(X_train_s, y_train)
            bucket.append(accuracy_score(y_test, clf.predict(X_test_s)))

    # 作图
    plt.figure(figsize=(9, 6))
    # CV 曲线（含误差带）
    plt.plot(unique_k, best_mean_per_k, color='#1f77b4', label='CV mean accuracy (best combo per K)')
    plt.fill_between(unique_k,
                     np.array(best_mean_per_k) - np.array(std_per_k),
                     np.array(best_mean_per_k) + np.array(std_per_k),
                     color='#1f77b4', alpha=0.15, label='CV ±1 std (per K)')
    # 测试集曲线（仅作参考，不用于选参）
    plt.plot(k_range_show, test_acc_uniform, 'o-', color='#2ca02c', label=f'Test (metric={metric}, weights=uniform)')
    plt.plot(k_range_show, test_acc_distance, 'o-', color='#d62728', label=f'Test (metric={metric}, weights=distance)')

    plt.xticks(range(1, 51, 2))
    plt.yticks(np.linspace(0.8, 1.0, 11))
    plt.ylim(0.85, 1.0)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('KNN validation curve and test accuracy')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 字体设置（避免 Unicode “−” 符号在 SimHei 下的缺字警告）
fm = FontManager()
_ = set(f.name for f in fm.ttflist)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    optimize_and_plot()