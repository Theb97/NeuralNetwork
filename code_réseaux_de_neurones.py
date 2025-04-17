# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 10:09:31 2025

@author: theob
"""
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork(object):

    def __init__(self, nombre_neurones_par_couche, entree, sortie_desiree):
        """nombre_neurones_par_couche est une liste contenant le nombre de neurone pour chaque rang de couche. Elle va de la couche d'entrée à la couche de sortie."""
        self.nombre_neurones_par_couche = nombre_neurones_par_couche
        self.x = entree
        self.y = np.array(sortie_desiree)
        self.activation_function_choice = int(input(
            "Choix possibles pour la fonction d'activation : ReLU ; Sigmoid ; GELU ; Soft_Plus. // Tapez 1 pour ReLU, 2 pour Sigmoïd, 3 pour GELU, 4 pour Soft_Plus."))
        self.omega_t = self.omega_init()
        self.bias_t = self.bias_init()
        
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feed_forward_test_data(x)), y)
                        for x, y in zip(test_data[0],test_data[1])]
        return sum(int(x == y) for (x, y) in test_results)

    def omega_init(self):

        n = len(self.nombre_neurones_par_couche)
        omega_init = []
        for k in range(n-1):
            omega = np.random.normal(
                0, 1, (self.nombre_neurones_par_couche[k+1], self.nombre_neurones_par_couche[k])).astype(np.float32)
            omega_init.append(omega)
        return omega_init

    def bias_init(self):

        n = len(self.nombre_neurones_par_couche)
        bias_init = []

        for k in range(n-1):
            bias = np.random.normal(
                1, 1, (self.nombre_neurones_par_couche[k+1], 1)).astype(np.float32)
            bias_init.append(bias)
        return bias_init

    def update_omega_and_bias(self, epochs, mini_batch=10, learning_rate=0.1,test_data = None):

        # epochs correspond au nombre de périodes où l'on va appliquer la retropropagation du gradient sur l'ensemble des données
        if test_data: n_test = len(test_data[0])
        compteur = 0
        Etapes = [0]
        Couts = [self.mean_cost_least_square_function()]
        n = len(self.x)
        liste = np.arange(n)
        for j in range(epochs):
            np.random.shuffle(liste)
            for k in range(0, n, mini_batch):
                L = self.mean_gradients_W_and_Bias_of_the_cost(
                    liste[k:k+mini_batch])
                gradients_W, gradients_Bias = L[0], L[1]
            
                # Mise à jour vectorisée : pour chaque couche, on met à jour poids et biais
                self.omega_t = [w - learning_rate * gw for w, gw in zip(self.omega_t, gradients_W)]
                self.bias_t = [b - learning_rate * gb for b, gb in zip(self.bias_t, gradients_Bias)]
            compteur += 1
            if test_data:
                print( "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
            Etapes.append(compteur)
            Couts.append(self.mean_cost_least_square_function())
        plt.plot(Etapes, Couts, color='blue')

    def feed_forward(self, qq): #qq est l'indice de la liste self.x contenant n=50,000 elements
        n = len(self.nombre_neurones_par_couche)
        omega_t = self.omega_t
        bias_t = self.bias_t
        z = self.x[qq]
        for k in range(n-1):
            z = np.dot(omega_t[k], z)+ bias_t[k] 
            z = self.activation_function(z)
        return z
    
    def feed_forward_test_data(self, x): #x est la valeur d'entrée un tableau array (784,1) pour les données de test
        n = len(self.nombre_neurones_par_couche)
        omega_t = self.omega_t
        bias_t = self.bias_t
        z = x
        for k in range(n-1):
            z = np.dot(omega_t[k], z)+ bias_t[k] 
            z = self.activation_function(z)
        return z

    def sorties_couches(self, qq): #qq est l'indice de la liste self.x contenant n=50,000 elements
        n = len(self.nombre_neurones_par_couche)
        valeurs_couches = []
        valeurs_couches.append(self.x[qq])

        milieu_couches = []

        omega_t = self.omega_t
        bias_t = self.bias_t

        for k in range(n-1):
            z = np.dot(omega_t[k], valeurs_couches[-1])+bias_t[k]
            milieu_couches.append(z)
            a = np.zeros_like(z)
            a = self.activation_function(z)
            valeurs_couches.append(a)
        L = [valeurs_couches, milieu_couches]
        return L

    def calcul_delta(self, qq): #qq est l'indice de la liste self.x ou de la liste self.y contenant n=50,000 elements
        n = len(self.nombre_neurones_par_couche)
        delta = []

        """"a et y sont deux listes de même taille"""
        y = self.y[qq]
        L = self.sorties_couches(qq)
        a = L[0][-1]
        z = L[1][-1]

        # initialisation à delta^{L}
        delta.insert(0, 2*(a-y)*self.activation_function_prime(z))
        zz = L[1]
        for j in range(2, n):
            z = zz[n-j-1]
            delta.insert(0, np.dot(np.transpose(
                self.omega_t[-j+1]), delta[-j+1])*self.activation_function_prime(z))
        return delta

    def gradients_W_of_the_cost(self, qq):  #qq est l'indice de la liste self.x contenant n=50,000 elements
        n = len(self.nombre_neurones_par_couche)
        gradients = []

        a = self.sorties_couches(qq)[0][:]
        calcul_delta = self.calcul_delta(qq)
        for k in range(1, n):
            delta = calcul_delta[k-1]
            gradient = np.dot(delta, np.transpose(a[k-1]))
            gradients.append(gradient)

        return gradients

    def gradients_Bias_of_the_cost(self, qq): #qq est l'indice de la liste self.x contenant n=50,000 elements
        return self.calcul_delta(qq)

    def mean_gradients_W_and_Bias_of_the_cost(self, L):

        # on calcule mean_gradients_W_and_Bias_of_the_cost par la méthode des mini-batch
        # L est une liste d'entiers k tous différents que l'on va utiliser pour récupérer le k-ième vecteur d'entrée x
        m = len(L)
        gradients_W = self.gradients_W_of_the_cost(L[0])
        gradients_Bias = self.gradients_Bias_of_the_cost(L[0])
        for qq in range(1, m):
            gradients_W += self.gradients_W_of_the_cost(L[qq])
            gradients_Bias += self.gradients_Bias_of_the_cost(L[qq])
            
        return [[g / m for g in gradients_W], [g / m for g in gradients_Bias]]

    def cost_least_square_function(self, qq): #qq est l'indice de la liste self.x contenant n=50,000 elements
        """"a et y sont deux listes de même taille"""
        y = self.y[qq]
        a = self.feed_forward(qq)
        n = len(y)

        somme = 0
        for k in range(n):
            somme += (a[k]-y[k])*(a[k]-y[k])
        return (somme)

    def mean_cost_least_square_function(self):
        n = len(self.y)
        somme_2 = 0
        for qq in range(n):
            somme_2 += self.cost_least_square_function(qq)
        return (somme_2/n)

    def activation_function(self, z):

        if self.activation_function_choice == 1:
            return self.activation_function_ReLU(z)
        if self.activation_function_choice == 2:
            return self.activation_function_sigmoid(z)
        if self.activation_function_choice == 3:
            return self.activation_function_GELU(z)
        if self.activation_function_choice == 4:
            return self.activation_function_soft_plus(z)

    def activation_function_prime(self, z):
        """dérivée de la fonction d'activation"""

        if self.activation_function_choice == 1:
            return self.activation_function_ReLU_prime(z)
        if self.activation_function_choice == 2:
            return self.activation_function_sigmoid(z)*(1-self.activation_function_sigmoid(z))
        if self.activation_function_choice == 3:
            return self.activation_function_GELU_prime(z)
        if self.activation_function_choice == 4:
            return self.activation_function_sigmoid(z)

    def activation_function_ReLU(self, z):
        """z est une liste de taille k où k dépend de la couche i du réseaux de neurones"""
        k = len(z)
        a = []
        for j in range(k):
            if z[j] < 0:
                a.append(0)
            else:
                a.append(z[j])
        return np.array(a)

    def activation_function_ReLU_prime(self, z):
        """z est une liste de taille k où k dépend de la couche i du réseaux de neurones"""
        k = len(z)
        a = []
        for j in range(k):
            if z[j] < 0:
                a.append(0)
            elif z[j] > 0:
                a.append(1)
            else:
                a.append(0.5)
        return np.array(a)

    def activation_function_sigmoid(self, z):
        """z est une liste de taille k où k dépend de la couche i du réseaux de neurones"""
        return 1/(1+np.exp(-z))

    def Gauss_error_function(self, x, n=1000):
        """"En utilisant la méthode de Simpson"""
        # On s'assure que n est pair pour appliquer Simpson correctement.
        if n % 2 == 0:
            n += 1

        # Pour les x négatifs, on utilise la propriété d'imparité
        sign = 1
        if x < 0:
            sign = -1
            x = -x

        somme = 0
        h = x/n

        # Application de la formule de Simpson
        somme += 1+np.exp(-x*x)
        for i in range(1, n):
            y = i*h
            if i % 2 == 1:
                coefficient = 4
            else:
                coefficient = 2
            somme += coefficient*np.exp(-y*y)
        integral = h/6*somme*sign
        return integral*2/np.sqrt(3.1415926535)

    def activation_function_GELU(self, z):
        """z est une liste de taille k où k dépend de la couche i du réseaux de neurones"""
        k = len(z)
        a = []
        for j in range(k):
            a.append(1/2*z[j]*(1+self.Gauss_error_function(z[j]/np.sqrt(2))))
        return np.array(a)

    def activation_function_GELU_prime(self, z):
        """z est une liste de taille k où k dépend de la couche i du réseaux de neurones"""
        k = len(z)
        a = []
        for j in range(k):
            a.append(1/2*(1+self.Gauss_error_function(z[j]/np.sqrt(
                2)))+z[j]*np.exp(-1/2*z[j]*z[j])/(np.sqrt(2*3.1415926535)))
        return np.array(a)

    def activation_function_soft_plus(self, z):
        """z est une liste de taille k où k dépend de la couche i du réseaux de neurones"""
        k = len(z)
        a = []
        for j in range(k):
            a.append(np.log(1+np.exp(z[j])))
        return np.array(a)
