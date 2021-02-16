from bibliotecaRedesNeuronales import *
import sys
import numpy as np
import math
import re
from matplotlib import pyplot as plt

class NeuronaAutoencoder(Neurona):

    def __init__(self, identificador, valor, tipo):
        super().__init__(identificador,valor,tipo)

    def funcion_activacion(self, entrada):

        # Correccion por si la entrada es muy grande o muy pequeña, para evitar overflow
        if entrada > 100:
            entrada = 100
        elif entrada < (-100):
            entrada = -100

        sigmoide_binaria = 2/(1 + np.exp(- entrada)) - 1

        return float(sigmoide_binaria)

    def derivada_funcion_activacion(self):

        sigmoide_binaria = self.funcion_activacion(self.get_entrada_neta())
        derivada_sigmoide_binaria = 0.5*(1 + sigmoide_binaria) * (1 - sigmoide_binaria)

        return float(derivada_sigmoide_binaria)

def leer_fichero(fichero_train, fichero_test):

    datos = []

    # Leemos los datos del fichero
    with open(fichero_train, "r") as file:
        n_atributos_entrada, n_clases_salida = file.readline().split(" ")
        for line in file:
            linea = line[:-1].split(" ")
            datos.append([float(i) for i in linea])

    for dato in datos:
        for i, valor in enumerate(dato):
            if dato[i] == 0.0:
                dato[i] = -1.0
    # Listas con atributos y clases para entrenar y validar
    atributos_train = []
    atributos_test = []
    clases_train = []
    clases_test = []

    datosEntrada_test = []

    for dato in datos:
        clases_train.append(dato[- int(n_clases_salida):])
        atributos_train.append(dato[:int(n_atributos_entrada)])


    with open(fichero_test, "r") as file:
        n_atributos_entrada, n_clases_salida = file.readline().split(" ")
        for line in file:
            linea = line[:-1].split(" ")
            datosEntrada_test.append([float(i) for i in linea])

    for dato in datosEntrada_test:
        for i, valor in enumerate(dato):
            if dato[i] == 0.0:
                dato[i] = -1.0

    for dato in datosEntrada_test:
        clases_test.append(dato[- int(n_clases_salida):])
        atributos_test.append(dato[:int(n_atributos_entrada)])



    return int(n_atributos_entrada), int(n_clases_salida), atributos_train ,atributos_test, clases_train, clases_test

def main():

    fichero_entrada_train = sys.argv[4]
    fichero_entrada_test = sys.argv[5]

    # Capturamos el numero de neuronas y capas ocultas
    lista_tam_capas_ocultas = sys.argv[1].split(",")

    # Leemos el fichero de entrada
    n_atributos_entrada, n_clases_salida, atributos_train ,atributos_test, clases_train, clases_test = leer_fichero(fichero_entrada_train, fichero_entrada_test)

    # Numero de epocas que vamos a tener
    num_epocas = int(sys.argv[2])

    # Creamos la Red Neuronal
    autoencoder = RedNeuronal("autoencoder", float(sys.argv[3]))
    
    # Creamos las capas ocultas
    for i, capa in enumerate(lista_tam_capas_ocultas):
        c = Capa(str(i + 1))

        # Creamos las neuronas de cada capa oculta
        for z in range(int(capa)):
            n = NeuronaAutoencoder("NeuronaOculta_" +str(z), 0, 1)
            
            # Añadimos la neurona a la capa
            c.addNeurona(n)

        autoencoder.addCapa(c)

    # Creamos las neuronas de entrada, una por atributo
    i = 0
    for i in range(n_atributos_entrada):
        n = NeuronaAutoencoder("NeuronaEntrada_" +str(i), 0, 0)
        n = NeuronaAutoencoder("NeuronaEntrada_" +str(i), 0, 0)

        # Añadimos a la lista de neuronas de entrada
        autoencoder.addNeuronaEntrada(n)

    # Creamos las neuronas de salida
    # En primer lugar creamos la capa de salida y la añadimos a la lista de capas
    capa_salida = Capa(len(lista_tam_capas_ocultas) + 1)
    autoencoder.addCapa(capa_salida)
    
    # Creamos las neuronas, una por clase de salida y las añadimos a la capa
    i = 0
    for i in range(n_clases_salida):
        n = NeuronaAutoencoder("NeuronaSalida_" +str(i), 0, 1)

        # Las añadimos a la lista
        autoencoder.capas[-1].addNeurona(n)
        
    # Enlaces la capa de entrada con la primera capa oculta
    for neurona_oculta in autoencoder.capas[0].neuronas:
        for neurona_entrada in autoencoder.neuronas_entrada:
            e = Enlace(np.random.uniform(-0.5,0.5), neurona_entrada, neurona_oculta)

            # Añadimos los enlaces a la primera capa oculta
            autoencoder.capas[0].addEnlace(e)

    # Creamos los enlaces de todas las neuronas, las de salida con la capa oculta, la capa oculta con 
    # la capa de salida
    i = 0
    for i, capa in enumerate(autoencoder.capas[:-1]):
        for neurona_destino in autoencoder.capas[i + 1].get_neuronas():
            for neurona_origen in capa.neuronas:
                e = Enlace(np.random.uniform(-0.5,0.5), neurona_origen, neurona_destino)

                # Añadimos el enlace a la capa correspondiente
                autoencoder.capas[i+1].addEnlace(e)

    PE = []
    MPE = []
    LRC = []

    # Comenzamos el entrenamiento
    for epoca in range(num_epocas):
        pe_epoca = 0
        lrc_epoca = 0

        # Bucle que recorre los datos de train(Paso 2)
        for i, dato in enumerate(atributos_train):
            # FeedForward
            # Bucle que recorre cada atributo del dato de entrenamiento en concreto(Paso 3)
            for ind, atributo in enumerate(dato):
                autoencoder.neuronas_entrada[ind].set_valor(float(atributo))


            # Calculamos las activaciones de las neuronas, capa a capa, y calculamos su valor, en funcion de la respuesta
            # de su funcion de activacion (Pasos 4 y 5)
            ind = 0
            for ind, capa in enumerate(autoencoder.capas):
                # Caso de la capa de entrada a la primera oculta, ya que no se considera como una capa como tal
                if ind == 0:
                    tam_capa_anterior = len(autoencoder.neuronas_entrada)

                autoencoder.entrada_neta_capa_a_capa(capa, tam_capa_anterior)
                capa.respuestas_activacion()

                # Necesario para el calculo de la entrada neta en nuestro diseño
                tam_capa_anterior = len(capa.neuronas)

            # Generamos una lista con los valores predichos por la capa de salida para compararlo con el real
            pixeles_activados = []
            for neurona in autoencoder.capas[-1].neuronas:
                if neurona.get_valor() > 0:
                    pixeles_activados.append(1.0)
                else:
                    pixeles_activados.append(-1.0)

            errores = np.equal(pixeles_activados, clases_train[i])
            pixeles_fallados = len(clases_train[i]) - np.count_nonzero(errores)
            pe_epoca += pixeles_fallados

            if pixeles_fallados == 0:
                lrc_epoca += 1

            # Backpropagation
            # Paso 6 -- Propagacion del error por parte de las neuronas de salida a la primera capa oculta.
            autoencoder.backpropagation_salida_a_capa_oculta(clases_train[i])

            # Paso 7 -- Backpropagation de las capas ocultas
            autoencoder.backpropagation_oculta_oculta()

            # Actualizacion de pesos y sesgos
            autoencoder.actualizacion_pesos()
        PE.append(pe_epoca)
        MPE.append(pe_epoca/len(clases_train))
        LRC.append(lrc_epoca)

    # Una vez terminado el entrenamiento hacemos la validacion
    i = 0
    PE_test = 0
    MPE_test = 0
    LRC_test = 0

    for i, dato_test in enumerate(atributos_test):
        # Establecemos los valores de las neuronas de entrada
        ind = 0
        for ind, atributo in enumerate(dato_test):
            autoencoder.neuronas_entrada[ind].set_valor(float(atributo))

        # Propagamos esos valores hasta obtener un salida
        ind = 0
        for ind, capa in enumerate(autoencoder.capas):

            if ind == 0:
                tam_capa_anterior = len(autoencoder.neuronas_entrada)

            autoencoder.entrada_neta_capa_a_capa(capa, tam_capa_anterior)
            capa.respuestas_activacion()

            tam_capa_anterior = len(capa.neuronas)

        # Calculamos las respuestas de la capa de salida
        capa_salida = autoencoder.capas[-1]
        valores_capa_salida = np.empty(len(capa_salida.neuronas))
        for j, neurona in enumerate(capa_salida.neuronas):
            valores_capa_salida[j] = neurona.get_valor()

        # Vemos si se produce acierto o no
        pixeles_activados = []
        for neurona in autoencoder.capas[-1].neuronas:
            if neurona.get_valor() > 0:
                pixeles_activados.append(1.0)
            else:
                pixeles_activados.append(-1.0)

        errores = np.equal(pixeles_activados, clases_test[i])
        pixeles_fallados = len(clases_test[i]) - np.count_nonzero(errores)
        PE_test += pixeles_fallados

        if pixeles_fallados == 0:
            LRC_test += 1

    MPE_test = PE_test/len(clases_test)

    print("PE: ", PE_test)
    print("LRC: ", LRC_test)
    print("MPE: ", MPE_test)

if __name__ == "__main__":
    main()
