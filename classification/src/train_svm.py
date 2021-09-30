"""
Klassifizieren der Seiten durch eine SVM:
"""

import config
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV




"""
Versuch #2
"""

"""Definiert und führt ein grid seach über gegebene Hyperparameter für rbf und polynomiellen Kernel durch
Jeder Lauf wird 5-fach cross validiert."""
def run_grid_search(X_digits, y_digits, C_range, degree_range, gamma_range):

    print(X_digits)
    print(type(X_digits))

    # Define grid over parameters for rbf and poly kernel
    rbf_grid = {"kernel": ["rbf"], "C": C_range, "gamma": gamma_range}
    poly_grid = {"kernel": ["poly"], "C": C_range, "degree": degree_range}

    # GridSearch arguments
    args = dict(
        estimator=svm.SVC(), scoring=make_scorer(accuracy_score), cv=5, verbose=1, n_jobs=-1
    )
    # Run gridsearch on the RBF grid
    print("GRID-SEARCH RBF")
    rbf_gs = GridSearchCV(param_grid=rbf_grid, **args).fit(X_digits, y_digits)
    print("GRID-SEARCH POLY")
    poly_gs = GridSearchCV(param_grid=poly_grid, **args).fit(X_digits, y_digits)

    print("Done with Gridsearch")
    return poly_gs, rbf_gs

def plot_support_vectors(X, y, clf, title=None, x_axis="", y_axis=""):
    
    # plot the line, the points, and the nearest vectors to the plane
    plt.figure()
    plt.clf()

    plt.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=90,
        facecolors="none",
        zorder=10,
        edgecolors="k",
    )

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolors="k")

    plt.axis("tight")

    x_min, x_max = X[:, 0].min() * 0.90, X[:, 0].max() * 1.1
    y_min, y_max = X[:, 1].min() * 0.90, X[:, 1].max() * 1.1

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired, shading='auto')
    plt.contour(
        XX,
        YY,
        Z,
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
        levels=[-1, 0, 1],
    )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    if title:
        plt.title(title)
    
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    # produce a legend with the unique colors from the scatter
    plt.legend(*scatter.legend_elements(),
                    loc="lower right", title="Klassen").set_zorder(102)
    plt.show()

def run_grid(df):
    import train_svm_utils as utils

    X_train = df[["covid_word_count","restriction_word_count","covid_word_count_url","restriction_word_count_url"]].values
    y_train = df[["label"]].values.ravel()

    # Define hyperparameter ranges
    gamma_range = [2 ** i for i in range(-13, -4)] # range(-13, -4)
    C_range = [2 ** i for i in range(-3, 4)] # range(-3, 4)
    degree_range = range(1, 7) # range(1, 7)

    poly_gs, rbf_gs = run_grid_search(X_train, y_train, C_range, degree_range, gamma_range)

    # Visualize the gridsearch results
    plt.figure(figsize=(8, 4))

    plt.suptitle("GridSearch Validation Accuracy")
    utils.visualize_gridsearch(
        poly_gs,
        C_range,
        degree_range,
        ylabel="C",
        xlabel="degree",
        title="SVM Poly Kernel",
        fignum=121,
    )
    plt.show()

    utils.visualize_gridsearch(
        rbf_gs,
        C_range,
        gamma_range,
        ylabel="C",
        xlabel="$\gamma$",
        title="SVM RBF Kernel",
        logx=True,
        fignum=122,
        cbar=True,
    )
    plt.show()

def run(df, fold):

    
    train_df = df[df.kfold != fold].reset_index(drop=True)
    print(f"Länge Traing_DF  {len(train_df)}")
    
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    print(f"Länge Valid_DF  {len(valid_df)}")

    X_train = train_df[["covid_word_count","restriction_word_count"]].values
    y_train = train_df[["label"]].values.ravel()
    
    X_valid = valid_df[["covid_word_count","restriction_word_count"]].values
    y_valid = valid_df[["label"]].values.ravel()

    # RBF Kernel hat sich für diesen Fall als Reinfall rausgestellt.
    #rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
    poly = svm.SVC(kernel='poly', degree=2, C=1).fit(X_train, y_train) # -3 , 4
    svc_model = svm.SVC(kernel='linear', C=1e10, max_iter=10000).fit(X_train, y_train)
    #svc_model = svm.SVC(kernel='poly', degree=1, C=1).fit(X_train, y_train)    

    poly_pred = poly.predict(X_valid)
    lin_pred = svc_model.predict(X_valid)

    # Berechnen der Accuracy und F1 Score der SVM mit polynomiellen Kernel
    poly_accuracy = accuracy_score(y_valid, poly_pred)
    poly_f1 = f1_score(y_valid, poly_pred, average='weighted')
    print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
    print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

    # Das gleiche mit linearem Kernel.
    rbf_accuracy = accuracy_score(y_valid, lin_pred)
    rbf_f1 = f1_score(y_valid, lin_pred, average='weighted')
    print('Accuracy (Linear Kernel): ', "%.2f" % (rbf_accuracy*100))
    print('F1 (Linear Kernel): ', "%.2f" % (rbf_f1*100))


    plot_support_vectors(
        X=X_train,
        y=y_train, 
        clf=poly,
        title="kernel='poly', degree=2, C=1",
        x_axis="covid_word_count",
        y_axis="restriction_word_count")
    plot_confusion_matrix(poly, X_valid, y_valid)

    plot_support_vectors(
        X=X_train,
        y=y_train, 
        clf=svc_model,
        title="kernel='linear', C=1e10, max_iter=10000",
        x_axis="covid_word_count",
        y_axis="restriction_word_count")
    plot_confusion_matrix(svc_model, X_valid, y_valid)
  

if (__name__ == "__main__"):

    # load data
    df = pd.read_csv(config.DATASET_PATH)

    # TODO: das muss auch noch schöner gemacht werden,
    # ist vorerst testweise
    df['link_impf_count'] = df['link'].apply(lambda x: str(x).count("impf"))
    df['body_impf_count'] = df['body'].apply(lambda x: str(x).count("impf"))
    df['meta_impf_count'] = df['meta_data'].apply(lambda x: str(x).count("impf"))
    df['title_impf_count'] = df['title'].apply(lambda x: str(x).count("impf"))
    

    df2 = df[df['label'] == 1]
    df = df.append(df2, ignore_index=True).reset_index()

    print("POSITIVE EINTRÄGE")
    print(len(df[df['label'] == 1]))
    print("NEGATIVE EINTRÄGE")
    print(len(df[df['label'] == 0]))

    # Testweise, zur zweidimensionalen Darstellung
    df['covid_word_count'] = df['covid_word_count'] + df['covid_word_count_url']
    #df['restriction_word_count'] = df['restriction_word_count'] + df['restriction_word_count_url']
    df['restriction_word_count'] = df['restriction_word_count'] + df['restriction_word_count_url'] + df['link_impf_count']  + df['body_impf_count']  + df['meta_impf_count']  + df['title_impf_count']

    # TODO: nur zur veranschaulichung, da manche weit drüber sind!
    df["covid_word_count"] = df["covid_word_count"].apply(lambda x: 100 if x > 100 else x)
    df["restriction_word_count"] = df["restriction_word_count"].apply(lambda x: 100 if x > 100 else x)

    # StratifiedKFold für eine Gleichverteilung der Label Werte
    kf = StratifiedKFold(n_splits=config.K_FOLD_SPLITS)

    # füllen den kfold Spalte
    for f, (t_, v_) in enumerate(kf.split(X=df, y=df.label.values)):
        df.loc[v_, 'kfold'] = f


    for i in range(config.K_FOLD_SPLITS):
        print(f"\n–––––––––––– FOLD {i} ––––––––––––\n")
        run(df, fold=i)
        