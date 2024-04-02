import numpy as np
from sklearn.linear_model import Lasso
from scenarios import scenario_1,scenario_2,scenario_3
import matplotlib.pyplot as plt

n_list = [100,200,400]
p = 50
s = 10
scenario = '3'
rho_list = [0.9]

sigma2 = 0.9
s=10
for n in n_list:
    for rho in rho_list:
        X,y = scenario_2(n,p,s,rho, sigma2,cant_clusters=10)
        lambda_max = np.linalg.norm(np.matmul(np.transpose(X), y), np.inf) / n
        eps = 0.001
        lambda_min = eps * lambda_max
        start = np.log10(lambda_min)
        end = np.log10(lambda_max)
        K = 100
        lambdas = np.logspace(start, end, K)
        coeficientes = []
        for lambda_ in lambdas:
            clf = Lasso(alpha=lambda_, fit_intercept=False)
            clf.fit(X, y)
            coeficientes.append(clf.coef_)
        fig, ax = plt.subplots()
        for k in range(p):
            if k<s:
                coef1 = [coeficientes[i][k] for i in range(K)]
                plt.plot(lambdas, coef1,color='green')
            else:
                coef1 = [coeficientes[i][k] for i in range(K)]
                plt.plot(lambdas, coef1, color='blue')
                title='n=%s'%(n)
                plt.title(title)
        plt.show()
