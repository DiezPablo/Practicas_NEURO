import numpy as np
import random
from abc import ABC, abstractmethod

class RedNeuronal():

	def __init__(self, identificador, constante_aprendizaje):
		self.identificador = identificador
		self.constante_aprendizaje = constante_aprendizaje
		self.enlaces = np.empty(0, dtype=object)

	def addEnlace(self, enlace):
		self.enlaces = np.append(self.enlaces, enlace)

	def get_enlaces(self):
		return self.enlaces

	def get_constante_aprendizaje(self):
		return self.constante_aprendizaje

	def calculo_suma_pesada(self):
		suma_pesada = 0
		for enlace in self.enlaces:
			valorNeuronaOrigen = float(enlace.get_neuronaOrigen().get_valor())
			pesoEnlace = enlace.get_peso()
			suma_pesada += valorNeuronaOrigen * pesoEnlace

		enlace.get_neuronaDestino().set_suma_pesada_entradas(suma_pesada)

class Enlace():

    def __init__(self, peso, neuronaOrigen, neuronaDestino):
        self.peso = peso
        self.neuronaOrigen = neuronaOrigen
        self.neuronaDestino = neuronaDestino

    def get_peso (self):
        return self.peso

    def get_neuronaOrigen(self):
        return self.neuronaOrigen

    def get_neuronaDestino (self):
        return self.neuronaDestino

    def set_peso (self, peso):
        self.peso = peso

    def set_neuronaOrigen (self, neuronaOrigen):
        self.neuronaOrigen = neuronaOrigen

    def set_neuronaDestino (self, neuronaDestino):
        self.neuronaDestino = neuronaDestino

class Neurona():
    def __init__(self, identificador, valor, tipo):
        self.identificador = identificador
        self.valor = valor
        self.tipo = tipo
        if self.tipo == 1:
        	self.suma_pesada_entradas = 0
        	self.sesgo = np.random.uniform(0.01,0.1)

    def get_identificador(self):
        return self.identificador

    def get_valor(self):
        return self.valor

    def get_sesgo(self):
        return self.sesgo

    def get_suma_pesada_entradas (self):
        return self.suma_pesada_entradas

    def set_sesgo(self, sesgo):
        self.sesgo = sesgo

    def set_valor(self, valor):
        self.valor = valor

    def set_suma_pesada_entradas(self, valor):
        self.suma_pesada_entradas = valor

    def set_identificador(self, identificador):
        self.identificador = identificador

    @abstractmethod
    def funcion_activacion(self, y_in):
        pass
