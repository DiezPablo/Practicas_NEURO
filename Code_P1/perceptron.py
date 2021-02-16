from bibliotecaRedesNeuronales import *
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


class NeuronaPerceptron(Neurona):

    def __init__(self, identificador, umbral, valor, tipo):
        super().__init__(identificador,valor,tipo)
        self.umbral = umbral

    def get_umbral(self):
        return self.umbral

    def set_umbral(self, umbral):
        self.umbral = umbral

    def funcion_activacion(self, y_in):

        if y_in > self.umbral:
            self.valor = 1
        elif y_in < -(self.umbral):
            self.valor = -1
        else:
            self.valor = 0

def leer_fichero(fichero):

    datos = []

    with open(fichero, "r") as file:
        n_atributos_entrada, n_clases_salida = file.readline().split(" ")
        for line in file:
            linea = line[:-1].split(" ")

            if linea[-2:] == ['1','0']:
                clase = -1
            else:
                clase = 1

            entrada = linea[:-2]
            entrada.append(clase)
            datos.append(entrada)

    return int(n_atributos_entrada), int(n_clases_salida), datos

def main():

    fichero_entrada = sys.argv[5]

    # Leemos el fichero de entrada
    n_atributos_entrada, n_clases_salida, datosEntrada = leer_fichero(fichero_entrada)

    # Capturamos el modo de funcionamiento
    modo_funcionamiento = int(sys.argv[1])


    if modo_funcionamiento == 3 and len(sys.argv) < 7:
        print("Para el modo de funcionamiento 3, es necesario indicar un fichero de test.")

    # Numero de epocas que vamos a tener
    num_epocas = int(sys.argv[2])

	# Creamos la red neuronal
    perceptron = RedNeuronal("Perceptron", float(sys.argv[3]))
    lista_neuronas_entrada = []

    for i in range(n_atributos_entrada):
        n = NeuronaPerceptron("NeuronaEntrada_" +str(i), 0, 0, 0)

        # Añadimos a la lista de neuronas de entrada
        lista_neuronas_entrada.append(n)

    # Creamos una unica neurona de salida, siempre 2 clases a predecir
    neurona_salida = NeuronaPerceptron("NeuronaSalida", float(sys.argv[4]), 0, 1)

    # Generamos todos los enlaces, de las neuronas de entrada, a las de salida
    for neurona in lista_neuronas_entrada:
        e = Enlace(0, neurona, neurona_salida)

        # Añadimos el enlace a la red neuronal
        perceptron.addEnlace(e)

    # Listas con atributos y clases para entrenar y validar
    atributos_train = []
    atributos_test = []
    clases_train = []
    clases_test = []

    # Definimos los distintos modos de Entrada
    if modo_funcionamiento == 1:

        porcentaje_train = 0.7

        # Generamos una lista de indices con todos los datos de entrada
        indices_aleat = np.random.permutation(len(datosEntrada))

        # Indices del dataset con los conjuntos de train y test
        indices_train = indices_aleat[:int(porcentaje_train * len(datosEntrada))]
        indices_test = indices_aleat[int(porcentaje_train*len(datosEntrada)):]

        # Extraemos los datos en concreto de esos indices para tenerlos listos.
        for i in indices_train:
            clases_train.append(datosEntrada[i].pop())
            atributos_train.append(datosEntrada[i])

        for i in indices_test:
            clases_test.append(datosEntrada[i].pop())
            atributos_test.append(datosEntrada[i])

    elif modo_funcionamiento == 2:

        for dato in datosEntrada:
            clase = dato.pop()
            clases_train.append(clase)
            clases_test.append(clase)
            atributos_train.append(dato)
            atributos_test.append(dato)

    elif modo_funcionamiento == 3:

        _, _, datosEntrada_test = leer_fichero(sys.argv[6])

        for dato in datosEntrada:
            clases_train.append(dato.pop())
            atributos_train.append(dato)

        for dato in datosEntrada_test:
            clases_test.append(dato.pop())
            atributos_test.append(dato)

        # Eliminamos la ruta de los argumentos
        fichero_entrada_disect = fichero_entrada.split("/")
        fichero_entrada_sin_extension = fichero_entrada_disect[-1].split(".")
        nombre_fichero_salida = fichero_entrada_sin_extension[0] +"_salida_perceptron.txt"
        fichero_salida = open(nombre_fichero_salida, "w")

    lista_errores_cuadraticos_epoca = np.empty(num_epocas)

    # Empieza el bucle de entrenamiento
    # Los pesos, sesgos y constante de constante_aprendizaje ya estan incializados
    i = 0
    # Recorremos todas las epocas
    for epoca in range(num_epocas):
        print("************** EPOCA ", epoca+1, "**************")

        lista_errores_cuadraticos_epoca[epoca] =0

        # Recorremos los datos de train
        for i, dato in enumerate(atributos_train):
            # Recorremos cada atributo de los datos de train

            for ind, atributo in enumerate(dato):
                # Recorremos la lista de entradas
                perceptron.enlaces[ind].get_neuronaOrigen().set_valor(atributo)

            # Calcular respuesta de la neurona de salida
            b = neurona_salida.get_sesgo()
            perceptron.calculo_suma_pesada()
            y_in = b + neurona_salida.get_suma_pesada_entradas()

            # Lo almacenamos para tener el error cuadratico medio de cada epoca
            error_cuadratico = (clases_train[i] - y_in)
            lista_errores_cuadraticos_epoca[epoca] += error_cuadratico

            # Obtenemos la clase predicha por la salida
            neurona_salida.funcion_activacion(y_in)
            clase_pred = neurona_salida.get_valor()

            # En caso de que prediga mal, ajustamos los pesos
            if clase_pred != clases_train[i]:

                # Ajuste de pesos de los enlaces
                for enlace in perceptron.enlaces:
                    peso_actual = enlace.get_peso()

                    peso_actualizado = peso_actual + (perceptron.get_constante_aprendizaje()*float(enlace.get_neuronaOrigen().get_valor())*clases_train[i])
                    enlace.set_peso(peso_actualizado)


                # Ajuste de peso del sesgo
                sesgo_actual = neurona_salida.get_sesgo()
                sesgo_actualizado = sesgo_actual + (perceptron.get_constante_aprendizaje() * clases_train[i])
                neurona_salida.set_sesgo(sesgo_actualizado)

        lista_errores_cuadraticos_epoca[epoca] = lista_errores_cuadraticos_epoca[epoca]/len(atributos_train)

    i = 0
    ind = 0
    clases_pred = np.empty(len(clases_test))

    # Validacion de los datos de test con los pesos finales de entrenamiento de la red
    for i, dato in enumerate(atributos_test):

        # Establecemos las activaciones
        for ind, atributo in enumerate(dato):
            perceptron.enlaces[ind].get_neuronaOrigen().set_valor(atributo)

        # Calculamos la salida de la red
        perceptron.calculo_suma_pesada()
        y_in = b + neurona_salida.get_suma_pesada_entradas()

        neurona_salida.funcion_activacion(y_in)

        # Guardamos la clase predicha en la lista de predicciones para posteriormente compararla con la real
        clase_predicha = neurona_salida.get_valor()
        clases_pred[i] = clase_predicha

    # Calculamos aciertos y errores
    aciertos = np.equal(clases_pred, clases_test)

    # Si el modo de funcionamiento es el 3 y se ha indicado fichero de test, escribimos fichero de salida
    i = 0
    if modo_funcionamiento == 3:
        linea_informativa = str(n_atributos_entrada) +" " +str(n_clases_salida) +"\n"
        fichero_salida.write(linea_informativa)
        for i, dato_test in enumerate(atributos_test):
            for atributo in dato_test:
                fichero_salida.write(str(atributo) +" ")
            if clases_pred[i] == -1:
                fichero_salida.write("1 0\n")
            else:
                fichero_salida.write("0 1\n")

        fichero_salida.close()

    # Calculamos el porcentaje de acierto
    porcentaje_acierto = aciertos.sum(axis=0)/len(clases_test)*100
    print("Porcentaje de acierto = ",porcentaje_acierto)

    # Calculamos la matriz de confusion
    matriz = confusion_matrix(clases_pred, clases_test)
    print("Matriz de confusion Perceptron:")
    print(matriz)


    # Plot del error cuadrativo medio por epoca
    plt.plot(lista_errores_cuadraticos_epoca, c='blue', label = 'Error cuadratico medio')
    plt.xlabel("Epocas")
    plt.ylabel("Error cuadratico medio")
    plt.title("Evolucion ECM del Perceptron")
    plt.show()

if __name__ == "__main__":
	main()
