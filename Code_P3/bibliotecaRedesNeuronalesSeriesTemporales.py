import numpy as np
import random
from abc import ABC, abstractmethod

class RedNeuronal():

    def __init__(self, identificador, constante_aprendizaje):
        self.identificador = identificador
        self.constante_aprendizaje = constante_aprendizaje
        self.capas = np.empty(0, dtype=object)
        self.neuronas_entrada = np.empty(0, dtype = object)
    
    def addCapa(self, capa):
        self.capas = np.append(self.capas, capa)

    def addNeuronaEntrada(self, neurona):
        self.neuronas_entrada = np.append(self.neuronas_entrada, neurona)

    def get_capas(self):
        return self.capas

    def get_constante_aprendizaje(self):
        return self.constante_aprendizaje

    def entrada_neta_capa_a_capa(self, capa, tam_capa_anterior):

        # Contador para moverse por las enlaces con misma neurona destino
        i = 0

        # Variable que almacena la acumulacion de la suma pesada de entradas
        suma_pesada_entradas = 0

        # Contador para asignarle el valor a la neurona destino especifica
        neurona_destino = 0

        for enlace in capa.enlaces:

            # Sumatorio de la suma pesada de entradas
            i += 1
            valor_neurona_origen = float(enlace.get_neuronaOrigen().get_valor())
            peso_enlace = enlace.get_peso()
            suma_pesada_entradas += valor_neurona_origen * peso_enlace

            # # Asignamos el valor a la neurona destino correspondiente
            if (i == tam_capa_anterior):

                # Calculamos la entrada neta de la neurona de la capa oculta
                entrada_neta = suma_pesada_entradas + enlace.get_neuronaDestino().get_sesgo()

                # Asignamos el valor a la variable entrada neta de la neurona destino correspondiente
                capa.neuronas[neurona_destino].set_entrada_neta(entrada_neta)

                # Calculamos el inicio de los enlaces donde empiezan los de la neurona de correspondiente que buscamos
                inicio_enlaces = neurona_destino * tam_capa_anterior

                # Calculo del final de la lista de enlaces
                final_enlaces = inicio_enlaces + tam_capa_anterior

                # Actualizamos el peso en todas los enlaces
                for x in range(inicio_enlaces, final_enlaces):
                    capa.enlaces[x].get_neuronaDestino().set_entrada_neta(entrada_neta)

                # Incremento del contador para en la siguiente iteracion pasar a la siguiente neurona de la capa con
                # todos los calculos hechos
                neurona_destino += 1

                # Restauramos los valores de los contadores, para pasar a la siguiente neurona destino
                i = 0
                suma_pesada_entradas = 0

    def backpropagation_salida_a_capa_oculta(self, clase_train):

        # En primer lugar obtenemos la capa de salida
        capa_salida = self.capas[-1]
        tam_capa_oculta = len(self.capas[-2].neuronas)

        # Lista que guardará el error para cada neurona de salida
        errores = np.empty(len(capa_salida.neuronas))

        # Pasamos las clases del entrenamiento a cada neurona de salida para calcular el error y las correciones
        i = 0
        for i, clase in enumerate(clase_train):

            # Calculo del error
            error = (float(clase) - capa_salida.neuronas[i].get_valor()) * capa_salida.neuronas[i].derivada_funcion_activacion_lineal()
            # error = (float(clase) - capa_salida.neuronas[i].get_valor()) * capa_salida.neuronas[i].derivada_funcion_activacion()

            # Guardamos el error
            errores[i] = error

        # Este contador sirve para movernos por la neurona destino de los distintos enlaces.
        neurona_salida = 0

        # Calculo de la correcion de pesos para cada enlace
        i = 0
        for enlace in capa_salida.enlaces:

            i += 1

            # Calculo de la correcion de pesos de cada enlace
            cte_learn = self.get_constante_aprendizaje()
            correcion_peso_enlace = cte_learn * errores[neurona_salida] * enlace.get_neuronaOrigen().get_valor()
            enlace.set_correcion_peso(correcion_peso_enlace)

            # Guardamos el error de los distintos enlaces, aunque se duplique, para tenerlo a la hora de la propagacion
            # a las distintas capas ocultas
            enlace.set_error(errores[neurona_salida])

            # Calculo de la correccion del sesgo
            correcion_sesgo = cte_learn * errores[neurona_salida]
            enlace.set_correcion_sesgo(correcion_sesgo)

            if i == tam_capa_oculta:
                neurona_salida += 1
                i = 0

    def backpropagation_oculta_oculta(self):

        # Neurona actual para la que estamos calculando
        neurona = 0

        capas_ocultas_inversas = list(reversed(self.capas))

        for i, capa in enumerate(capas_ocultas_inversas):

            # Condicion de parada, ya hemos terminado la retroprogacion
            if i == len(capas_ocultas_inversas) - 1:
                break

            # Control de si la capa a retropropagar es la primera oculta
            if (i == len(capas_ocultas_inversas) - 2):
                long_capa_siguiente = len(self.neuronas_entrada)
            else:
                long_capa_siguiente = len(capas_ocultas_inversas[i + 2].neuronas)

            # Numero de neuronas de la capa anterior, se utiliza en los calculos
            num_neuronas_capa_anterior = len(capas_ocultas_inversas[i+1].neuronas)

            # Array que guarda el delta_in para cada neurona de la capa
            delta_in = np.zeros(len(capas_ocultas_inversas[i+1].neuronas), dtype='f')

            # Contadores que indican a la neurona que le corresponde el producto y calculan cuando los enlaces
            # pertenecen a otra neurona
            neurona = 0
            contador_neurona = 0

            # Recorremos todos los enlaces para calcular el delta_in de cada neurona
            for enlace in capa.enlaces:

                # Acumulamos el contador, para saber cuando se terminan los enlaces de una determinada neurona
                contador_neurona += 1

                # Acumulacion de la multiplicacion del peso con el error del enlace
                delta_in[neurona] += enlace.get_peso() * enlace.get_error()

                # Permite cambiar de neurona destino en la acumulacion de los deltas
                if contador_neurona == len(capa.neuronas):

                    contador_neurona = 0
                    neurona += 1


            # Calculo de los errores utilizando el delta_in calculado anteriormente
            # Creamos un array que guarda los errores
            errores = np.empty(num_neuronas_capa_anterior,dtype = 'f')
            for x, neurona in enumerate(capas_ocultas_inversas[i+1].neuronas):
                if (i == len(capas_ocultas_inversas) - 2):
                    errores[x] = delta_in[x] * neurona.derivada_funcion_activacion()

                else:
                    errores[x] = delta_in[x] * neurona.derivada_funcion_activacion_lineal()
                # errores[x] = delta_in[x] * neurona.derivada_funcion_activacion()

            # Calculo de la correcion de peso y sesgo, que van en los enlaces
            neurona_salida = 0
            j = 0
            for enlace in capas_ocultas_inversas[i+1].enlaces:

                j += 1

                # Calculo de la correcion de pesos de cada enlace
                cte_learn = self.get_constante_aprendizaje()

                correcion_peso_enlace = cte_learn * errores[neurona_salida] * enlace.get_neuronaOrigen().get_valor()
                enlace.set_correcion_peso(correcion_peso_enlace)

                # Guardamos el error de los distintos enlaces, aunque se duplique, para tenerlo a la hora de la propagacion
                # a las distintas capas ocultas
                enlace.set_error(errores[neurona_salida])

                # Calculo de la correccion del sesgo
                correcion_sesgo = cte_learn * errores[neurona_salida]
                enlace.set_correcion_sesgo(correcion_sesgo)

                if j == long_capa_siguiente:
                    neurona_salida += 1
                    j = 0

    def actualizacion_pesos(self):

        for i,capa in enumerate(self.capas):

            for j,enlace in enumerate(capa.get_enlaces()):

                # Actualizamos el peso del enlace
                peso_actualizado = enlace.get_peso() + enlace.get_correcion_peso()
                enlace.set_peso(peso_actualizado)

                # Actualizamos le sesgo
                sesgo_actualizado = enlace.get_neuronaDestino().get_sesgo() + enlace.get_correcion_sesgo()
                enlace.get_neuronaDestino().set_sesgo(sesgo_actualizado)




class Enlace():

    def __init__(self, peso, neuronaOrigen, neuronaDestino):
        self.peso = peso
        self.neuronaOrigen = neuronaOrigen
        self.neuronaDestino = neuronaDestino
        self.correcion_peso = 0
        self.correcion_sesgo = 0
        self.error = 0

    def get_error(self):
        return self.error

    def get_correcion_sesgo(self):
        return self.correcion_sesgo

    def get_correcion_peso(self):
        return self.correcion_peso

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

    def set_correcion_peso(self, valor):
        self.correcion_peso = valor

    def set_correcion_sesgo(self, valor):
        self.correcion_sesgo = valor

    def set_error(self, valor):
        self.error = valor

class Capa():

    def __init__(self, numero_capa):
        self.numero_capa = numero_capa
        self.enlaces = np.empty(0, dtype=object)
        self.neuronas = np.empty(0, dtype = object)


    def respuestas_activacion(self):
        
        # Calculamos la respuesta de activacion de cada neruona
        for neurona in self.neuronas:

            # Obtenemos la entrada neta
            entrada_neta = neurona.get_entrada_neta()

            # Le pasamos la entrada neta a la funcion de activacion
            if self.numero_capa == '1':
                valor_activacion = neurona.funcion_activacion(entrada_neta)
            else:
                valor_activacion = neurona.funcion_activacion_lineal(entrada_neta)

            # El valor obtenido por la funcion de activacion se lo asignamos al valor de salida de la neurona
            neurona.set_valor(valor_activacion)


    def addNeurona(self, neurona):
        self.neuronas = np.append(self.neuronas, neurona)

    def addEnlace(self, enlace):
        self.enlaces = np.append(self.enlaces, enlace)

    def get_numero_capa(self):
        return self.numero_capa

    def get_tipo_capa(self):
        return self.tipo_capa

    def get_enlaces (self):
        return self.enlaces

    def get_neuronas(self):
        return self.neuronas

    def set_numero_capa(self, numero_capa):
        self.numero_capa = numero_capa

    def set_tipo_capa(self, tipo_capa):
        self.tipo_capa = tipo_capa



class Neurona():
    def __init__(self, identificador, valor, tipo):
        self.identificador = identificador
        self.valor = valor
        self.tipo = tipo
        if self.tipo != 0:
            self.entrada_neta = 0
            self.sesgo = np.random.uniform(0.01,0.1)

    def get_identificador(self):
        return self.identificador

    def get_valor(self):
        return self.valor

    def get_sesgo(self):
        return self.sesgo

    def get_entrada_neta(self):
        return self.entrada_neta

    def set_sesgo(self, sesgo):
        self.sesgo = sesgo

    def set_valor(self, valor):
        self.valor = valor

    def set_entrada_neta(self, valor):
        self.entrada_neta = valor

    def set_identificador(self, identificador):
        self.identificador = identificador

    @abstractmethod
    def funcion_activacion(self, y_in):
        pass
