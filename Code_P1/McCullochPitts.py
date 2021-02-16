import numpy as np

class RedMcCullochPitts():

	def __init__(self, identificador):
		self.identificador = identificador
		self.enlaces = np.empty(0, dtype=object)

	def addEnlace(self, enlace):
		self.enlaces = np.append(self.enlaces, enlace)

	def get_enlaces(self):
		return self.enlaces
		

class NeuronaMcCullochPits():

	def __init__(self, identificador, umbral = 0, valor = 0):
		self.identificador = identificador
		self.umbral = umbral
		self.valor = valor
		self.suma_propagacion = 0

	def get_identificador(self):
		return self.identificador

	def get_umbral(self):
	    return self.umbral

	def get_valor(self):
	    return self.valor

	def get_suma_propagacion (self):
		return self.suma_propagacion

	def set_valor(self, valor):
	    self.valor = valor

	def set_suma_propagacion(self, valor):
		self.suma_propagacion += valor
		
	def set_identificador(self, identificador):
	    self.identificador = identificador

	def set_umbral(self, umbral):
	    self.umbral = umbral

	def funcion_activacion(self):
		if self.suma_propagacion < self.umbral:
			self.valor = 0

			# Reseteamos el valor de la suma de propagacion para proximas iteraciones
			self.suma_propagacion = 0
		else:
			self.valor = 1

			# Reseteamos el valor de la suma de propagacion para proximas iteraciones
			self.suma_propagacion = 0

class Enlace:

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