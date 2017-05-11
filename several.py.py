import numpy as np
import matplotlib.pyplot as plt

import random as rd
import pandas as pd

##############Traitement#########################
def traitement(data_init):
    n, m = np.shape(data_init)
    result = np.zeros((0,m))
    for i in range(n):
        string = "?"
        if string not in data_init[i,:]:
            result = np.vstack((result, data_init[i,:]))
    result = np.array(result, dtype = float)
    return result

data_init = pd.read_csv("mammographic_masses.data",header=None)
data_init = np.array(data_init)

clean_data = traitement(data_init) #n=830,t=5
verite = clean_data[:,-1] #n = 830
donnees = clean_data[:,:-1]

### Signification de chaque caracteristique, dans l'ordre ###
# BI RADS
# Age de la patiente
# Forme de la masse, non ordinal
# Frontière de la masse, non ordinal 
# Densité, ordinal
# Bénigne (0) ou maline (1)

##################################################

def p(z, y, x, w, gamma, alpha, beta):
    """vecteur de taille n"""
    return p_y_xy(z, y, x, w, gamma).prod(1)*p_z_x(x, alpha, beta)

def p_z_x(x, alpha, beta):
    """vecteur de taille n"""
    return 1/(1+np.exp(-x.dot(alpha) - beta))
    
def eta(x, w, gamma):
    """matrice de taille n,t"""
    gamma = gamma.reshape(1,-1)
    return 1/(1+np.exp(-x.dot(w) - gamma))

def p_y_xy(z, y, x, w, gamma):
    """matrice de taille n,t"""
    expos = np.abs(y.T - z).T
    return ((1. - eta(x, w, gamma))**expos) * ((eta(x, w, gamma) ** (1-expos)))

def df_eta(y, x, w, gamma, alpha, beta):
    """Attention : w doit être un vecteur de taille d, y un vecteur de taille n et gamma un entier ! Retourne un vecteur de taille n"""
    print(w.shape)
    n = x.shape[0]                                          
    ZEROS = np.zeros(n)
    ONES = np.ones(n)                                          
    return ((-1)**eta(x, w, gamma)* (p(ONES, y, x, w, gamma, alpha, beta) - p(ZEROS, y, x, w, gamma, alpha, beta)))

def deta_w(y, x, w, gamma):
    """"matrice n,d lorsque tout sauf x est pris en t   """                                       
    return (x.T * eta(x, w, gamma) * (1-eta(x, w, gamma))).T
    
def df_w(y, x, w, gamma, alpha, beta):
    """matrice de taille d,T"""
    T = y.shape[1]
    d = x.shape[1]
    L=np.zeros((d,T))                                          
    for j in range(T):
        L[:,j] = df_eta(y[:,j], x, w[:,j], gamma[j], alpha, beta).dot( deta_w( y[:,j], x, w[:,j], gamma[j]) )
    return L

def deta_gamma(x, w, gamma):
    """Attention : w doit être un vecteur de taille d, y un vecteur de taille n et gamma un entier ! Retourne un vecteur de taille n"""
    return eta(x, w, gamma) * (1-eta(x, w, gamma))

def df_gamma(y, x, w, gamma, alpha, beta):
    """liste de taille t"""
    T = y.shape[1]
    L=np.zeros(T)                                          
    for j in range(T):
        print(deta_gamma(x, w[:,j], gamma[j]))
        print(df_eta(y[:,j], x, w[:,j], gamma[j], alpha, beta))
        L[j] = deta_gamma(x, w[:,j], gamma[j])[0].dot( df_eta(y[:,j], x, w[:,j], gamma[j], alpha, beta)[0] )
    return L


def grad_alpha(y, x, w, gamma, alpha, beta):
    """dérivée de f_obj par alpha : vecteur de taille d"""
    n = x.shape[0]                                          
    ZEROS = np.zeros(n)
    ONES = np.ones(n)
    Delta_p = p(ONES, y, x, w, gamma, alpha, beta) - p(ZEROS, y, x, w, gamma, alpha, beta)
    return x.T.dot(Delta_p*(np.exp(-np.dot(x,alpha) - beta))/(1+np.exp(-np.dot(x,alpha) - beta))**2)

def grad_beta(y, x, w, gamma, alpha, beta):
    """dérivée de f_obj par beta : scalaire"""
    n = x.shape[0]  
    ZEROS = np.zeros(n)
    ONES = np.ones(n)
    Delta_p = p(ONES, y, x, w, gamma, alpha, beta) - p(ZEROS, y, x, w, gamma, alpha, beta)
    return np.dot( Delta_p , ( np.exp(-np.dot(x, alpha) - beta) ) / ( (1+np.exp(-np.dot(x, alpha) - beta))**2 ) )

def BFGS_alpha(y, data, w, gamma, alpha, beta, grad, nb_iter = 100, eps = 0.01):
    #Initialisation
    d = alpha.shape[0]
    W = np.eye(d)
    I = np.eye(d)
    k = 0
    ONES = np.ones((d,d))
    x_new = alpha
    delta_x = I.dot(x_new)
    delta_g = grad(y, data, w, gamma, x_new, beta)

    #Boucle
    while k <nb_iter or np.linalg.norm(delta_x) < eps:
        denom = (delta_g.dot(delta_x))
        W = (I-I*delta_x.dot(ONES.dot(delta_g.T))/denom ).dot(W).dot(I-I*delta_g.dot(ONES.dot(delta_x.T))/denom ) + I*delta_x.dot(ONES.dot(delta_x.T))/denom
        d = - W.dot(grad(y, data, w, gamma, x_new, beta))
        x, x_new = x_new, x_new + d
        delta_x = I.dot(x_new - x)
        delta_g = grad(y, data, w, gamma, x_new, beta) - grad(y, data, w, gamma, x, beta)
        k += 1
    print(k)
    return x_new


def EM_algorithm(annotations, donnees, verite, nb_iter = 50):
    n,d = donnees.shape
    n,t = annotations.shape
    
    #initialisaition
    alpha = np.zeros(d)
    alpha_news = np.ones(d)
    beta = 0
    beta_news = 1
    w = np.ones(d,t)
    gamma = np.ones(t)
    epsilon = 0.01
    
    while np.linalg.norm(alpha-alpha_news) + (beta - beta_new)**2 > espilon :
        #E-step
        P_z = p(verite, annotations, donnees, w, gamma, alpha_new, beta_new)
        #M-step
        alpha, beta = alpha_new, beta_new #mise à jour d'alpha et beta

        """BFGS"""
        alpha_new = BFGS_alpha()
        return(f)
        
        
X = np.array([[1,2,3],[4,3,6],[8,5,6],[4,5,6],[4,5,6]])
W = np.array([[-5,5,6,6],[4,0,28,52],[4,5,8,5]])
Gamma = np.array([1,2,3,5])
Y = np.array([[1,0,1,0],[0,1,0,1],[0,0,0,1],[1,1,0,1],[1,1,0,1]])
Z = np.array([1,0,1,1,0])
Alpha = np.array([6,8,9])
Beta= 8

U = p_z_x(X, Alpha, Beta)
