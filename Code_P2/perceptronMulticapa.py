from bibliotecaRedesNeuronales import *
import sys
import numpy as np
import math
import re
from matplotlib import pyplot as plt

class NeuronaPerceptronMulticapa(Neurona):

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
def matriz_confusion(pred, real, n_clases):

    # Tranformamos los vectores a valore entero, para poder indexar directamente la funcion
    vector_transformado_clases_test = transformacion_vector(np.asarray(real,dtype=np.float64),n_clases)
    vector_transformado_clases_pred = transformacion_vector(np.asarray(pred,dtype=np.float64),n_clases)

    matriz = np.zeros((n_clases,n_clases))

    for i,j in zip (vector_transformado_clases_pred, vector_transformado_clases_test):
        matriz[i][j] += 1


    print(matriz)
def transformacion_vector(vector, n_clases):

    # Esta funcion transforma el vector a valores numericos crecientes para facilitar la generacion de la
    # matriz de confusion
    vector_tranformado = np.empty(len(vector),dtype='int64')
    for i, clase in enumerate(vector):
        if n_clases == 2:
            if (clase[0] == 0) and (clase[1] == 1):
                vector_tranformado[i] = 0
            else:
                vector_tranformado[i] = 1
        else:
            if (clase[0] == 0) and (clase[1] == 0) and (clase[2] == 1):
                vector_tranformado[i] = 0
            elif (clase[0] == 0) and (clase[1] == 1) and (clase[2] == 0):
                vector_tranformado[i] = 1
            else:
                vector_tranformado[i] = 2

    return vector_tranformado


def normalizacion_estandar(datos_train, datos_test, n_atributos):

    # Calculo de la media
    means = np.zeros(n_atributos)

    for dato in datos_train:
        for i, atributo in enumerate(dato):
            means[i] += float(atributo)

    n_datos = len(datos_train)
    for i,mean in enumerate(means):
        means[i] = mean / n_datos

    # Calculo de la desviacion estandar
    stds = np.zeros(n_atributos)
    for dato in datos_train:
        for i, atributo in enumerate(dato):
            stds[i] += (float(atributo) - means[i])**2

    for i,std in enumerate(stds):
        stds[i] = math.sqrt(std / n_datos)

    # Ahora normalizamos los datos de train utilizando la formula (X - media)/desv. estandar
    for dato_train in datos_train:
        for i, atributo in enumerate(dato_train):
            dato_train[i] = (float(atributo) - means[i])/stds[i]

    # Los datos de test se normalizan utilizando la media y desviacion tipica de los de train
    for dato_test in datos_test:
        for i, atributo in enumerate(dato_test):
            dato_test[i] = (float(atributo) - means[i])/stds[i]

    return


def leer_fichero(fichero_entrada, modo_funcionamiento, porcentaje):

    datos = []

    # Leemos los datos del fichero
    with open(fichero_entrada, "r") as file:
        n_atributos_entrada, n_clases_salida = file.readline().split(" ")
        for line in file:
            linea = line[:-1].split(" ")
            datos.append([float(i) for i in linea])


    # Listas con atributos y clases para entrenar y validar
    atributos_train = []
    atributos_test = []
    clases_train = []
    clases_test = []

    if modo_funcionamiento == 1:

        porcentaje_train = porcentaje /100

        # Generamos una lista de indices con todos los datos de entrada
        indices_aleat = np.random.permutation(len(datos))

        # Indices del dataset con los conjuntos de train y test
        indices_train = indices_aleat[:int(porcentaje_train * len(datos))]
        indices_test = indices_aleat[int(porcentaje_train*len(datos)):]

        # Extraemos los datos en concreto de esos indices para tenerlos listos.
        for i in indices_train:
            clases_train.append(datos[i][- int(n_clases_salida):])
            atributos_train.append(datos[i][:int(n_atributos_entrada)])

        for i in indices_test:
            clases_test.append(datos[i][- int(n_clases_salida):])
            atributos_test.append(datos[i][:int(n_atributos_entrada)])

    elif modo_funcionamiento == 2:

        for dato in datos:    
            clase = dato[- int(n_clases_salida):]
            clases_train.append(clase)
            clases_test.append(clase)
            atributos_train.append(dato[:int(n_atributos_entrada)])
            atributos_test.append(dato[:int(n_atributos_entrada)])

    elif modo_funcionamiento == 3:

        datosEntrada_test = []

        for dato in datos:
            clases_train.append(dato[- int(n_clases_salida):])
            atributos_train.append(dato[:int(n_atributos_entrada)])

        with open(sys.argv[8], "r") as file:
            n_atributos_entrada, n_clases_salida = file.readline().split(" ")
            for line in file:
                linea = line[:-1].split(" ")
                datosEntrada_test.append([float(i) for i in linea])

        for dato in datosEntrada_test:
            clases_test.append(dato[- int(n_clases_salida):])
            atributos_test.append(dato[:int(n_atributos_entrada)])

    return int(n_atributos_entrada), int(n_clases_salida), atributos_train ,atributos_test, clases_train, clases_test

def main():

    fichero_entrada = sys.argv[7]

    # Capturamos el modo de funcionamiento
    modo_funcionamiento = int(sys.argv[1])

    # Capturamos el numero de neuronas y capas ocultas
    lista_tam_capas_ocultas = sys.argv[2].split(",")

    porcentaje_particion_train = [i for i in re.split(r'(\d+)', sys.argv[5]) if i]

    # Leemos el fichero de entrada
    n_atributos_entrada, n_clases_salida, atributos_train ,atributos_test, clases_train, clases_test = leer_fichero(fichero_entrada, modo_funcionamiento, 70)

    # Normalizacion
    if sys.argv[6] == "True":
        normalizacion_estandar(atributos_train, atributos_test, n_atributos_entrada)

    if modo_funcionamiento == 3 and len(sys.argv) < 9:
        print("Para el modo de funcionamiento 3, es necesario indicar un fichero de test.")

    # Numero de epocas que vamos a tener
    num_epocas = int(sys.argv[3])

    # Creamos la Red Neuronal
    perceptronMulticapa = RedNeuronal("PerceptronMulticapa", float(sys.argv[4]))
    
    # Creamos las capas ocultas
    for i, capa in enumerate(lista_tam_capas_ocultas):
        c = Capa(str(i + 1))

        # Creamos las neuronas de cada capa oculta
        for z in range(int(capa)):
            n = NeuronaPerceptronMulticapa("NeuronaOculta_" +str(z), 0, 1)
            
            # Añadimos la neurona a la capa
            c.addNeurona(n)

        perceptronMulticapa.addCapa(c)

    # Creamos las neuronas de entrada, una por atributo
    i = 0
    for i in range(n_atributos_entrada):
        n = NeuronaPerceptronMulticapa("NeuronaEntrada_" +str(i), 0, 0)

        # Añadimos a la lista de neuronas de entrada
        perceptronMulticapa.addNeuronaEntrada(n)

    # Creamos las neuronas de salida
    # En primer lugar creamos la capa de salida y la añadimos a la lista de capas
    capa_salida = Capa(len(lista_tam_capas_ocultas) + 1)
    perceptronMulticapa.addCapa(capa_salida)
    
    # Creamos las neuronas, una por clase de salida y las añadimos a la capa
    i = 0
    for i in range(n_clases_salida):
        n = NeuronaPerceptronMulticapa("NeuronaSalida_" +str(i), 0, 1)

        # Las añadimos a la lista
        perceptronMulticapa.capas[-1].addNeurona(n)
        
    # Enlaces la capa de entrada con la primera capa oculta
    for neurona_oculta in perceptronMulticapa.capas[0].neuronas:
        for neurona_entrada in perceptronMulticapa.neuronas_entrada:
            e = Enlace(np.random.uniform(-0.5,0.5), neurona_entrada, neurona_oculta)

            # Añadimos los enlaces a la primera capa oculta
            perceptronMulticapa.capas[0].addEnlace(e)

    # Creamos los enlaces de todas las neuronas, las de salida con la capa oculta, la capa oculta con 
    # la capa de salida
    i = 0
    for i, capa in enumerate(perceptronMulticapa.capas[:-1]):
        for neurona_destino in perceptronMulticapa.capas[i + 1].get_neuronas():
            for neurona_origen in capa.neuronas:
                e = Enlace(np.random.uniform(-0.5,0.5), neurona_origen, neurona_destino)

                # Añadimos el enlace a la capa correspondiente
                perceptronMulticapa.capas[i+1].addEnlace(e)

    # Creamos la lista de errores cuadraticos medios por epoca y numero de aciertos
    lista_errores_cuadraticos_epoca = np.empty(num_epocas)
    porcentaje_acierto_train = np.empty(num_epocas)

    # Comenzamos el entrenamiento
    for epoca in range(num_epocas):

        # Inicializamos el valor que calcula el error cuadratico medio de la epoca
        lista_errores_cuadraticos_epoca[epoca] = 0

        # Inicializamos el acierto de train de la epoca
        porcentaje_acierto_train[epoca] = 0

        # Lista de predicciones del train
        valores_pred_train = np.empty((len(clases_train),n_clases_salida))

        # Bucle que recorre los datos de train(Paso 2)
        for i, dato in enumerate(atributos_train):
            # FeedForward
            # Bucle que recorre cada atributo del dato de entrenamiento en concreto(Paso 3)
            for ind, atributo in enumerate(dato):
                perceptronMulticapa.neuronas_entrada[ind].set_valor(float(atributo))


            # Calculamos las activaciones de las neuronas, capa a capa, y calculamos su valor, en funcion de la respuesta
            # de su funcion de activacion (Pasos 4 y 5)
            ind = 0
            for ind, capa in enumerate(perceptronMulticapa.capas):
                # Caso de la capa de entrada a la primera oculta, ya que no se considera como una capa como tal
                if ind == 0:
                    tam_capa_anterior = len(perceptronMulticapa.neuronas_entrada)

                perceptronMulticapa.entrada_neta_capa_a_capa(capa, tam_capa_anterior)
                capa.respuestas_activacion()

                # Necesario para el calculo de la entrada neta en nuestro diseño
                tam_capa_anterior = len(capa.neuronas)

            # Antes de retropropagar el error calculamos el ECM
            valores_capa_salida = [neurona.get_valor() for neurona in perceptronMulticapa.capas[-1].neuronas]
            error_cuadratico = 0
            for j in range(len(valores_capa_salida)):
                resta_cuadrados = clases_train[i][j] - valores_capa_salida[j]
                error_cuadratico += resta_cuadrados**2

            error_cuadratico = error_cuadratico/n_clases_salida

            # Lo añadimos a la lista de errores cuadraticos medios
            lista_errores_cuadraticos_epoca[epoca] += error_cuadratico

            # Modificamos el array a notacion binaria
            ind_max = np.argmax(valores_capa_salida, axis=0)
            valores_capa_salida = [0 for valor in valores_capa_salida]
            valores_capa_salida[ind_max] = 1

            valores_pred_train[i] = valores_capa_salida


            # Backpropagation
            # Paso 6 -- Propagacion del error por parte de las neuronas de salida a la primera capa oculta.
            perceptronMulticapa.backpropagation_salida_a_capa_oculta(clases_train[i])

            # Paso 7 -- Backpropagation de las capas ocultas
            perceptronMulticapa.backpropagation_oculta_oculta()

            # Actualizacion de pesos y sesgos
            perceptronMulticapa.actualizacion_pesos()

        # Calculo del ECM para la epoca
        lista_errores_cuadraticos_epoca[epoca] = lista_errores_cuadraticos_epoca[epoca] / len(atributos_train)

        # Calculo del porc. de aciertos para la epoca
        resultado = np.equal(valores_pred_train, np.asarray(clases_train[i], dtype=np.float64))
        aciertos = np.sum(resultado[:, 0], axis=0)

        porcentaje_acierto_train[epoca]= aciertos / len(resultado[:, 0])

    # Una vez terminado el entrenamiento hacemos la validacion
    i = 0
    clases_pred = np.empty((len(clases_test),n_clases_salida))
    for i, dato_test in enumerate(atributos_test):
        # Establecemos los valores de las neuronas de entrada
        ind = 0
        for ind, atributo in enumerate(dato_test):
            perceptronMulticapa.neuronas_entrada[ind].set_valor(float(atributo))

        # Propagamos esos valores hasta obtener un salida
        ind = 0
        for ind, capa in enumerate(perceptronMulticapa.capas):

            if ind == 0:
                tam_capa_anterior = len(perceptronMulticapa.neuronas_entrada)

            perceptronMulticapa.entrada_neta_capa_a_capa(capa, tam_capa_anterior)
            capa.respuestas_activacion()

            tam_capa_anterior = len(capa.neuronas)

        # Calculamos las respuestas de la capa de salida
        capa_salida = perceptronMulticapa.capas[-1]
        valores_capa_salida = np.empty(len(capa_salida.neuronas))
        for j, neurona in enumerate(capa_salida.neuronas):
            valores_capa_salida[j] = neurona.get_valor()

        # Convertimos esas respuestas a binario, para ello, el valor mayor de todas las neuronas de salida es el que
        # se pone a 1, el resto a 0, no consideramos posible empates
        ind_max = np.argmax(valores_capa_salida,axis = 0)
        valores_capa_salida.fill(0)
        valores_capa_salida[ind_max] = 1
        clases_pred[i] = valores_capa_salida


    resultado = np.equal(clases_pred, np.asarray(clases_test,dtype=np.float64))
    aciertos = np.sum(resultado[:,0],axis=0)
    print("Porcentaje de acierto: ", aciertos/len(resultado[:,0]))



    i = 0
    if modo_funcionamiento == 3:
        fichero_salida = open("prediccion_mlp.txt", "w")
        linea_informativa = str(n_atributos_entrada) +" " +str(n_clases_salida) +"\n"
        fichero_salida.write(linea_informativa)
        for i, dato_test in enumerate(atributos_test):
            for atributo in dato_test:
                fichero_salida.write(str(atributo) +" ")
            for j in range(len(clases_pred[i])):
                fichero_salida.write(str(int(clases_pred[i][j])))
                if j != len(clases_pred[i]) -1:
                    fichero_salida.write(" ")

            fichero_salida.write("\n")

        fichero_salida.close()

    # Plot del error cuadrativo medio por epoca
    # plt.plot(lista_errores_cuadraticos_epoca, c='blue', label = 'Error cuadratico medio')
    # plt.xlabel("Epocas")
    # plt.ylabel("Error cuadratico medio")
    # plt.title("Evolucion ECM - XOR - CteLearn 0.1 - 5000 epocas")
    # plt.show()

    # n = "ECM_n_epocas_" + str(num_epocas) + " _" + fichero_entrada.split("/")[-1] + "_" + "cteLearn_" + sys.argv[4] +"num_capas_" +str(len(lista_tam_capas_ocultas))+"_" +sys.argv[6] +".txt"
    # f = open(n, "w")
    # for error in lista_errores_cuadraticos_epoca:
    #     f.write(str(error) +",")
    # f.write(str(aciertos/len(resultado[:,0])))
    # f.close()

    print(" ---- Informacion del problema ejecutado  ----")
    print(" - Modo de funcionamiento: ", modo_funcionamiento)
    if modo_funcionamiento == 1:
        print(" - Porcentaje de train: ", porcentaje_particion_train[0], porcentaje_particion_train[1])
    print(" - Dataset1: ", fichero_entrada)
    if modo_funcionamiento == 3:
        print(" - Dataset de test:",sys.argv[8])
    print(" - Epocas de entrenamiento: ", num_epocas)
    print(" - Constante de aprendizaje: ", sys.argv[4])
    print(" - Numero neuronas capas ocultas: ", lista_tam_capas_ocultas)
    print(" - Normalizacion: ", sys.argv[6])

    # Calculo de la matriz de confusion
    print("---- Matriz de confusion ----")
    matriz_confusion(clases_pred, clases_test, n_clases_salida)



if __name__ == "__main__":
    main()
