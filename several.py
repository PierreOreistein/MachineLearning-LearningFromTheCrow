import numpy as np
import matplotlib.pyplot as plt

import random as rd
import math
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
    res = p_y_xy(z, y, x, w, gamma)
    res = np.where(res > 10e-7, res, 10e-7)
    return res.prod(1)*p_z_x(z,x, alpha, beta)

def p_z_x(z,x, alpha, beta):
    """vecteur de taille n"""
    n = x.shape[0]
    ZEROS = np.zeros(n)
    ONES = np.ones(n)
    #return 1/(1+np.exp(-np.dot(x,alpha) - beta))
    return - z + (2*z - ONES)*(1/(1+np.exp(-np.dot(x,alpha) - beta))) + ONES #+ 1*np.ones(x.shape[0])

def eta(x, w, gamma):
    """matrice de taille n,t"""
    gamma = gamma.reshape(1,-1)
    n = x.shape[0]
    #print("Yo")
    #print(np.dot(x,w) - gamma )
    return 1/(1+np.exp(-np.dot(x,w) - gamma )) #+ 1*np.ones(gamma.shape)
def p_y_xy(z, y, x, w, gamma):
    """matrice de taille n,t"""
    expos = np.abs(y.T - z).T
    #print('p_y')
    #print(((1. - eta(x, w, gamma))**expos) * ((eta(x, w, gamma) ** (1-expos))))
    return ((1. - eta(x, w, gamma))**expos) * ((eta(x, w, gamma) ** (1-expos)))

def df_eta(y, x, w, gamma, alpha, beta):
    """Attention : w doit être un vecteur de taille d, y un vecteur de taille n et gamma un entier ! Retourne un vecteur de taille n"""
    n = x.shape[0]                                          
    ZEROS = np.zeros(n)
    ONES = np.ones(n)
    return (-1)**y * (2*p(ONES, y, x, w, gamma, alpha, beta) - p(ZEROS, y, x, w, gamma, alpha, beta))

def deta_w(y, x, w, gamma):
    """"matrice n,d lorsque tout sauf x est pris en t   """                                       
    return (x.T * eta(x, w, gamma) * (1-eta(x, w, gamma))).T
    
def df_w(y, x, w, gamma, alpha, beta):
    """matrice de taille d,T"""
    T = y.shape[1]
    d = x.shape[1]
    L=np.zeros((d,T))                                          
    for j in range(T):
        L[:,j] = np.dot(df_eta(y[:,j], x, w[:,j], gamma[j], alpha, beta), deta_w( y[:,j], x, w[:,j], gamma[j]) )
    return L

def deta_gamma(x, w, gamma):
    """Attention : w doit être un vecteur de taille d, y un vecteur de taille n et gamma un entier ! Retourne un vecteur de taille n"""
    return eta(x, w, gamma) * (1-eta(x, w, gamma))

def df_gamma(y, x, w, gamma, alpha, beta):
    """vecteur de taille t"""
    T = y.shape[1]
    L=np.zeros(T)
    for j in range(T):
        #print('df_eta')
        #print(deta_gamma(x, w[:,j], gamma[j]))
        L[j] = np.dot(deta_gamma(x, w[:,j], gamma[j]), df_eta(y[:,j], x, w[:,j], gamma[j], alpha, beta) )
    return np.array(L)


def grad_alpha(y, x, w, gamma, alpha, beta):
    """dérivée de f_obj par alpha : vecteur de taille d"""
    n = x.shape[0]                                          
    ZEROS = np.zeros(n)
    ONES = np.ones(n)
    Delta_p = p(ONES, y, x, w, gamma, alpha, beta) - p(ZEROS, y, x, w, gamma, alpha, beta)
    """
    print("Exponentielle")
    print(-np.dot(x,alpha) - beta)
    print("Exp")
    print (1+np.exp(-np.dot(x,alpha) - beta))
    print("X")
    print(x)
    print("alpha")
    print(alpha)
    print("beta")
    print(beta)
    """
    return np.dot(x.T, (Delta_p*(np.exp(-np.dot(x,alpha) - beta))/(1+np.exp(-np.dot(x,alpha) - beta))**2) )

def grad_beta(y, x, w, gamma, alpha, beta):
    """dérivée de f_obj par beta : scalaire"""
    n = x.shape[0]  
    ZEROS = np.zeros(n)
    ONES = np.ones(n)
    Delta_p = p(ONES, y, x, w, gamma, alpha, beta) - p(ZEROS, y, x, w, gamma, alpha, beta)
    return np.dot( Delta_p , ( np.exp(-np.dot(x, alpha) - beta) ) / ( (1+np.exp(-np.dot(x, alpha) - beta))**2 ) )

def grad(y, data, Vect):
    d = data.shape[1]
    t = y.shape[1]
    alpha = Vect[:d]
    beta = Vect[d]
    gamma = Vect[d+1:d+1+t]
    w = Vect[d+1+t:].reshape((d,t))
    D_Alpha = grad_alpha(y, data, w, gamma, alpha, beta).reshape(-1)
    D_Beta = grad_beta(y, data, w, gamma, alpha, beta).reshape(-1)
    D_gamma = df_gamma(y, data, w, gamma, alpha, beta).reshape(-1) #nan
    D_w = df_w(y, data, w, gamma, alpha, beta).reshape(-1) #nan
    """print("grad")
    print(D_Alpha)
    print(D_Beta)
    print(D_gamma)
    print(D_w)"""
    return np.concatenate((D_Alpha, D_Beta, D_gamma, D_w))

Hessien = []

def Hess(delta_x, delta_g, H):                                                   
    size = delta_x.shape[0]
    I = np.eye(size)
    ONES = np.ones((size, size))
    denom = np.dot(delta_g, delta_x)
    D1 = np.dot( I*delta_x , np.dot(ONES, delta_g.T) ) / denom
    D2 = np.dot( I*delta_g , np.dot(ONES, delta_x.T) ) / denom
    D3 = np.dot( I*delta_x , np.dot(ONES, delta_x.T) ) / denom
    return np.dot( np.dot( (I - D1) , H), I - D2) + D3
    
def BFGS(y, data, w, gamma, alpha, beta, grad, nb_iter = 100, eps = 0.000001):
    #Initialisation
    print(alpha)
    print(beta)
    print(gamma)
    print(w)
    Vect = np.concatenate((alpha.reshape((-1)), beta.reshape((-1)), gamma.reshape((-1)), w.reshape((-1))))
    size = Vect.shape[0]
    print('Vect')
    print(Vect)
    H = np.eye(size)
    k = 0
    
    x_new = Vect
    delta_x = x_new
    delta_g = grad(y, data, x_new)
    print('delta_x')
    print(delta_x)
    print(delta_g)
    #Boucle
    while k < nb_iter and np.linalg.norm(delta_x) > eps:
        
        H = Hess(delta_x, delta_g, H)
        print(H.shape)
        print("H")
        print(H)
        d = np.dot(H,grad(y, data, x_new))
        print("d")
        print(d)
        x, x_new = x_new, x_new + d # x_new = nan
        delta_x = x_new - x
        delta_g = grad(y, data, x_new) - grad(y, data, x)
        k += 1
        print("x_new")
        print(x_new)
        #print("iter")
        
    print("Vect arrivée")
    print(x_new)
    d = data.shape[1]
    t = y.shape[1]
    alpha = x_new[:d]
    beta = np.array([x_new[d]])
    gamma = x_new[d+1:d+1+t]
    w = x_new[d+1+t:].reshape((d,t))
    print("Il y a eu " + str(k) + " itérations.")
    print(alpha,beta,gamma,w)
    return (alpha, beta, gamma, w)


def EM_algorithm(donnees, annotations, nb_iteration = 50, epsilon = 0.00001):
    n,d = donnees.shape
    n,t = annotations.shape
    
    #initialisaition
    alpha = np.zeros(d)
    alpha_new = (1e-5)*np.random.rand()*np.ones(d)
    beta = np.array([0])
    beta_new = (1e-5)*np.random.rand()*np.array([1])
    w = (1e-5)*np.random.rand()*np.ones((d,t))
    gamma = (1e-5)*np.random.rand()*np.ones(t)
    i = 0

    n = donnees.shape[0]                                          
    ZEROS = np.zeros(n)
    ONES = np.ones(n)
    #print(p(ONES , annotations, donnees, w, gamma, alpha_new, beta_new))# + p(ONES , annotations, donnees, w, gamma, alpha_new, beta_new) )
    #(alpha_new, beta_new, gamma, w) = BFGS(annotations, donnees, w, gamma, alpha_new, beta_new, grad)
    
    while np.linalg.norm(alpha-alpha_new) + (beta - beta_new)**2 > epsilon and i < nb_iteration:
        #E-step
        P_z = p(verite, annotations, donnees, w, gamma, alpha_new, beta_new)
        #M-step
        alpha, beta = alpha_new, beta_new #mise à jour
        (alpha_new, beta_new, gamma, w) = BFGS(annotations, donnees, w, gamma, alpha, beta, grad)
        i += 1
    
    return (alpha_new, beta_new, gamma, w)
        

def doc_decide(alpha, beta, label):
    if label == 0:
        return int(rd.random() > beta)
    else:
        return int(rd.random() < alpha)
    
def choix_medecin(alphas, betas, labels):
    # Simulation du choix de chaque radiologue
    nb_medecin = len(alphas)
    nb_patient = np.shape(labels)[0]
    data = np.array([[doc_decide(alphas[j], betas[j], labels[i]) for j in range(nb_medecin) ] for i in range(nb_patient)])
    return data

"""        
M = np.array([[1,2,3,4],[1,1,3,8],[1,6,3,5],[3,2,9,8]])        
X = np.array([[1,2,3],[4,3,6],[8,5,6],[4,5,6],[4,5,6]])
W = (1e-5)*np.array([[5,5,6,6],[4,0,2,52],[4,5,8,5]])
Gamma = (1e-5)*np.array([1,2,-3,-5])
Y = np.array([[1,0,1,0],[0,1,0,1],[0,0,0,1],[1,1,0,1],[1,1,0,1]])
Z = np.array([1,0,1,1,0])
Alpha = (1e-5) *np.array([6,-8,9])
Beta= (1e-5)*np.array(8)

U = BFGS(Y, X, W, Gamma, Alpha, Beta, grad)
"""

# Spécificite et sensibilite des annotateurs 
alphas = np.array([0.9, 0.8, 0.57, 0.6, 0.55])
betas = np.array([0.95, 0.85, 0.62, 0.65, 0.58])

# Données pour l'apprentissage
data = clean_data[:,:-1]
labels = np.array(clean_data[:,5])
annotations = choix_medecin(alphas, betas, labels)

# Valeur initiale
data_line_nb, data_col_nb = np.shape(data)

# Optimisation
learnt_alphas, learnt_beta, learnt_gamma, learnt_w = EM_algorithm(donnees=data, annotations=annotations)
"""
# Affichage des résultats
print('Alpha')
print("d'origine : ", alphas)
print("calculé   : ", learnt_alphas)
print('Max des erreurs sur les alpha_j : ', np.max(abs(alphas - learnt_alphas)))
print('\n')

print('Beta : ')
print("d'origine : ", betas)
print("calculé   : ", learnt_beta)
#print('Max des erreurs sur les beta_j : ', np.max(abs(betas-learnt_beta)))
#print('\n')

print('Erreur label : ', np.sum(labels), np.sum(mu))

"""
