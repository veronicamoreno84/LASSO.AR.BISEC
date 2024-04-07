
import numpy as np
from sklearn.preprocessing import RobustScaler
from LASSO_areas import area_tray_coef_lasso
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def area_tray_coef_lasso(X,y,fit_intercept=False):
    '''
    Considera una grilla de valores de lambdas y aplica LASSO en todos los lambda de la grilla
    :param X: Features
    :param y: target
    :return: El area bajo la curva del coeficiente, normalizado (para que sume uno el vector de las areas)
    '''
    n,p = X.shape

    lambdas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    areas = p * [0]
    for i in range(len(lambdas)-1):
        lambda_ = lambdas[i+1]
        clf = LogisticRegression(penalty='l1',C=lambda_,fit_intercept=fit_intercept,solver='saga')
        clf.fit(X, y)
        coeficientes = clf.coef_
        print(coeficientes.shape)
        for j in range(p):
            areas[j] += abs(coeficientes[1,j])*(lambda_-lambdas[i])
    norm = [area / sum(areas) for area in areas]
    print(norm)
    return norm

def grafico_areas_ordenadas(X,y,fit_intercept=False, name=False,savefig=False,showfig=True,save_in = None):
    '''


    :param X:
    :param y:
    :param fit_intercept:
    :param name:
    :param savefig:
    :param showfig:
    :param save_in:
    :return: Las variable ordenadas según las áreas
    '''


    n, p = X.shape

    areas = area_tray_coef_lasso(X, y, fit_intercept=fit_intercept)
    indices_ordenan_areas = np.argsort(areas).tolist()
    indices_ordenan_areas.reverse()
    areas_reversed= [areas[i] for i in indices_ordenan_areas]
    fig, ax = plt.subplots()
    tau_list=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for tau in tau_list:
        plt.axhline(y=tau, color='black', linestyle='--',
                    linewidth=1)
    plt.plot([i for i in range(p)], areas_reversed, 'co', color='blue', )
    title = r'Ordered areas'
    if name:
        title+=name

    plt.title(title)
    if savefig:
        filename = '\ %s.png' % (name)
        if save_in is not None:
            filename = save_in + filename
        plt.savefig(fname=filename)
    if showfig:
        plt.show()

    return indices_ordenan_areas


from sklearn import datasets

X, y = datasets.load_breast_cancer(return_X_y=True)

rb = RobustScaler()
X = rb.fit_transform(X)
print(X.shape)
print(y.shape)

grafico_areas_ordenadas(X,y,fit_intercept=True, name=False,savefig=False,showfig=True,save_in = None)