import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

input_data = pd.read_csv("dataset/raw_data.csv").values
colors = ["green", "blue", "orange", "red", "yellow", "cyan", "magenta", "purple"]

tsne = TSNE(n_components=2, perplexity=100)
scaler = preprocessing.StandardScaler()


def time_converter(str_time):
    return sum(x * int(t) for x, t in zip([3600, 60, 1], str_time.split(":")))


def friendly_format_parser():
    for i in range(len(input_data)):
        input_data[i][1] = time_converter(input_data[i][1])
        input_data[i][2] = input_data[i][2].replace("%", "")
        input_data[i][2] = float("{0:.4f}".format(float(input_data[i][2]) * 0.01))


def get_scaled_data():
    X = input_data[:, :-1]
    y = input_data[:, -1]
    X = scaler.fit_transform(X)
    return X, y


def clean_y_data(y, n):
    res = []
    for i in range(len(y)):
        try:
            cur_y = int(y[i])
        except:
            cur_y = 0
        if n == 2:
            if cur_y > 0:
                cur_y = 1
        res.append(cur_y)
    return res


def plot_target_data(X_data, y_data):
    exist_labels = []
    plt.figure(figsize=(15, 15))
    for i in range(len(y_data)):
        id = int(y_data[i])
        if id in exist_labels:
            plt.scatter(X_data[i, 0], X_data[i, 1], c=colors[id], marker=".", alpha=0.3)
        else:
            plt.scatter(X_data[i, 0], X_data[i, 1], c=colors[id], marker=".", label=("RevLevel_" + str(id)), alpha=0.3)
            exist_labels.append(id)
    plt.legend()
    plt.show()


def print_model_info(y_true, y_pred, message):
    size = len(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print(message)
    print("accuracy = %.1f" % (accuracy * 100.0) + " %. TestDataSet size =", str(size))
    print("TN = %.1f; FP = %.1f; FN = %.1f; TP = %.1f\n" %
          (tn * 100.0 / size, fp * 100.0 / size, fn * 100.0 / size, tp * 100.0 / size))
    print()


# n_classes = 2, 5
def run_regression(n_classes):
    X, y = get_scaled_data()
    y = clean_y_data(y, n_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = LinearRegression().fit(X_train, y_train)

    y_pred = model.predict(X_test)
    for i in range(len(y_pred)):
        y_pred[i] = int(y_pred[i])
    print_model_info(y_test, y_pred, ("Run LinearRegression for", n_classes, "classes:"))


# n_clusters = 2, 5 or 8
def run_clustering(n_clusters):
    X, y = get_scaled_data()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    y_pred = kmeans.predict(X)

    if n_clusters < 8:
        y = clean_y_data(y, n_clusters)
        print_model_info(y, y_pred, ("Run K-Means clustering for", n_clusters, "clusters:"))
    X = tsne.fit_transform(X)
    plot_target_data(X, y_pred)


if __name__ == '__main__':
    friendly_format_parser()

    #run_regression(2)
    #run_regression(5)
    #run_clustering(2)
    #run_clustering(5)
    run_clustering(8)
