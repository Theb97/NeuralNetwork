# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 12:00:01 2025

@author: theob
"""

import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

entree_training_data = training_data[0]
sortie_desiree_training_data = training_data[1]

entree_validation_data = validation_data[0]
sortie_desiree_validation_data = validation_data[1]

entree_test_data = test_data[0]
sortie_desiree_test_data = test_data[1]

import code_réseaux_de_neurones

net = code_réseaux_de_neurones.NeuralNetwork([784,30,30,10],entree_training_data,sortie_desiree_training_data)

net.update_omega_and_bias(30,10,0.5,test_data)
