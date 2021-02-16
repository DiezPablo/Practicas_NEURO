from bibliotecaRedesNeuronalesSeriesTemporales import *
import sys
import numpy as np
import math
import re
from matplotlib import pyplot as plt


class NeuronaSerieTemporal(Neurona):

    def __init__(self, identificador, valor, tipo):
        super().__init__(identificador, valor, tipo)

    def funcion_activacion(self, entrada):

        # Correccion por si la entrada es muy grande o muy pequeña, para evitar overflow
        if entrada > 100:
            entrada = 100
        elif entrada < (-100):
            entrada = -100

        sigmoide_binaria = 2 / (1 + np.exp(- entrada)) - 1

        return float(sigmoide_binaria)

    def derivada_funcion_activacion(self):

        sigmoide_binaria = self.funcion_activacion(self.get_entrada_neta())
        derivada_sigmoide_binaria = 0.5 * (1 + sigmoide_binaria) * (1 - sigmoide_binaria)

        return float(derivada_sigmoide_binaria)

    def funcion_activacion_lineal(self, entrada):

        return entrada

    def derivada_funcion_activacion_lineal(self):

        return 1

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
    datos_completos = []

    if modo_funcionamiento == 1:

        porcentaje_train =float(porcentaje) / 100

        # Generamos una lista de indices con todos los datos de entrada
        indices = np.arange(len(datos))

        # Indices del dataset con los conjuntos de train y test
        indices_train = indices[:int(porcentaje_train * len(datos))]
        indices_test = indices[int(porcentaje_train * len(datos)):]

        # Extraemos los datos en concreto de esos indices para tenerlos listos.
        for i in indices_train:
            clases_train.append(datos[i][- int(n_clases_salida):])
            atributos_train.append(datos[i][:int(n_atributos_entrada)])
            datos_completos.append(datos[i][:int(n_atributos_entrada)])

        for i in indices_test:
            clases_test.append(datos[i][- int(n_clases_salida):])
            atributos_test.append(datos[i][:int(n_atributos_entrada)])
            datos_completos.append(datos[i][:int(n_atributos_entrada)])

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

        with open(sys.argv[7], "r") as file:
            n_atributos_entrada, n_clases_salida = file.readline().split(" ")
            for line in file:
                linea = line[:-1].split(" ")
                datosEntrada_test.append([float(i) for i in linea])

        for dato in datosEntrada_test:
            clases_test.append(dato[- int(n_clases_salida):])
            atributos_test.append(dato[:int(n_atributos_entrada)])

    return datos_completos, int(n_atributos_entrada), int(n_clases_salida), atributos_train, atributos_test, clases_train, clases_test


def main():
    fichero_entrada = sys.argv[6]

    # Capturamos el modo de funcionamiento
    modo_funcionamiento = int(sys.argv[1])

    # Capturamos el numero de neuronas y capas ocultas
    lista_tam_capas_ocultas = sys.argv[2].split(",")

    porcentaje_particion_train = [i for i in re.split(r'(\d+)', sys.argv[5]) if i]

    # Leemos el fichero de entrada
    datos, n_atributos_entrada, n_clases_salida, atributos_train, atributos_test, clases_train, clases_test = leer_fichero(
        fichero_entrada, modo_funcionamiento, porcentaje_particion_train[0])

    if modo_funcionamiento == 3 and len(sys.argv) < 8:
        print("Para el modo de funcionamiento 3, es necesario indicar un fichero de test.")

    # Numero de epocas que vamos a tener
    num_epocas = int(sys.argv[3])

    # Creamos la Red Neuronal
    seriesTemporal = RedNeuronal("SerieTemporal", float(sys.argv[4]))

    # Creamos las capas ocultas
    for i, capa in enumerate(lista_tam_capas_ocultas):
        c = Capa(str(i + 1))

        # Creamos las neuronas de cada capa oculta
        for z in range(int(capa)):
            n = NeuronaSerieTemporal("NeuronaOculta_" + str(z), 0, 1)

            # Añadimos la neurona a la capa
            c.addNeurona(n)

        seriesTemporal.addCapa(c)

    # Creamos las neuronas de entrada, una por atributo
    i = 0
    for i in range(n_atributos_entrada):
        n = NeuronaSerieTemporal("NeuronaEntrada_" + str(i), 0, 0)

        # Añadimos a la lista de neuronas de entrada
        seriesTemporal.addNeuronaEntrada(n)

    # Creamos las neuronas de salida
    # En primer lugar creamos la capa de salida y la añadimos a la lista de capas
    capa_salida = Capa(len(lista_tam_capas_ocultas) + 1)
    seriesTemporal.addCapa(capa_salida)

    # Creamos las neuronas, una por clase de salida y las añadimos a la capa
    i = 0
    for i in range(n_clases_salida):
        n = NeuronaSerieTemporal("NeuronaSalida_" + str(i), 0, 1)

        # Las añadimos a la lista
        seriesTemporal.capas[-1].addNeurona(n)

    # Enlaces la capa de entrada con la primera capa oculta
    for neurona_oculta in seriesTemporal.capas[0].neuronas:
        for neurona_entrada in seriesTemporal.neuronas_entrada:
            e = Enlace(np.random.uniform(-0.5, 0.5), neurona_entrada, neurona_oculta)

            # Añadimos los enlaces a la primera capa oculta
            seriesTemporal.capas[0].addEnlace(e)

    # Creamos los enlaces de todas las neuronas, las de salida con la capa oculta, la capa oculta con 
    # la capa de salida
    i = 0
    for i, capa in enumerate(seriesTemporal.capas[:-1]):
        for neurona_destino in seriesTemporal.capas[i + 1].get_neuronas():
            for neurona_origen in capa.neuronas:
                e = Enlace(np.random.uniform(-0.5, 0.5), neurona_origen, neurona_destino)

                # Añadimos el enlace a la capa correspondiente
                seriesTemporal.capas[i + 1].addEnlace(e)


    # Calculo del ECM basico
    # Para el calculo del ECM basico cogemos el último elemento de cada dato de train y lo predecimos, sobre eso calculamos
    # el ECM, siempre vamos a tener una nuerona de salida
    # Test
    ECM_basico = 0
    for dato in datos:
        pred_ECM = dato[-1]
        for atributo in dato:
            dif_dato_pred = atributo - pred_ECM
            ECM_basico += dif_dato_pred**2
    ECM_basico = ECM_basico / len(datos)

    # Creamos la lista de errores cuadraticos medios por epoca y numero de aciertos
    lista_errores_cuadraticos_epoca = np.empty(num_epocas)

    # Comenzamos el entrenamiento
    for epoca in range(num_epocas):

        # Inicializamos el valor que calcula el error cuadratico medio de la epoca
        lista_errores_cuadraticos_epoca[epoca] = 0

        # Bucle que recorre los datos de train(Paso 2)
        for i, dato in enumerate(atributos_train):
            # FeedForward
            # Bucle que recorre cada atributo del dato de entrenamiento en concreto(Paso 3)
            for ind, atributo in enumerate(dato):
                seriesTemporal.neuronas_entrada[ind].set_valor(float(atributo))

            # Calculamos las activaciones de las neuronas, capa a capa, y calculamos su valor, en funcion de la respuesta
            # de su funcion de activacion (Pasos 4 y 5)
            ind = 0
            for ind, capa in enumerate(seriesTemporal.capas):
                # Caso de la capa de entrada a la primera oculta, ya que no se considera como una capa como tal
                if ind == 0:
                    tam_capa_anterior = len(seriesTemporal.neuronas_entrada)

                seriesTemporal.entrada_neta_capa_a_capa(capa, tam_capa_anterior)
                capa.respuestas_activacion()

                # Necesario para el calculo de la entrada neta en nuestro diseño
                tam_capa_anterior = len(capa.neuronas)

            valores_capa_salida = [neurona.get_valor() for neurona in seriesTemporal.capas[-1].neuronas]

            error_cuadratico = 0
            for j in range(len(valores_capa_salida)):
                resta_cuadrados = clases_train[i][j] - valores_capa_salida[j]
                error_cuadratico += resta_cuadrados ** 2

            error_cuadratico = error_cuadratico / n_clases_salida

            # Lo añadimos a la lista de errores cuadraticos medios
            lista_errores_cuadraticos_epoca[epoca] += error_cuadratico

            # Backpropagation
            # Paso 6 -- Propagacion del error por parte de las neuronas de salida a la primera capa oculta.
            seriesTemporal.backpropagation_salida_a_capa_oculta(clases_train[i])

            # Paso 7 -- Backpropagation de las capas ocultas
            seriesTemporal.backpropagation_oculta_oculta()

            # Actualizacion de pesos y sesgos
            seriesTemporal.actualizacion_pesos()

        # Calculo del ECM para la epoca
        lista_errores_cuadraticos_epoca[epoca] = lista_errores_cuadraticos_epoca[epoca] / len(atributos_train)

    # Una vez terminado el entrenamiento hacemos la validacion
    i = 0
    valores_salida = []
    ecm_test = 0
    for i, dato_test in enumerate(atributos_test):

        # Establecemos los valores de las neuronas de entrada
        ind = 0
        for ind, atributo in enumerate(dato_test):
            seriesTemporal.neuronas_entrada[ind].set_valor(float(atributo))

        # Propagamos esos valores hasta obtener un salida
        ind = 0
        for ind, capa in enumerate(seriesTemporal.capas):

            if ind == 0:
                tam_capa_anterior = len(seriesTemporal.neuronas_entrada)

            seriesTemporal.entrada_neta_capa_a_capa(capa, tam_capa_anterior)
            capa.respuestas_activacion()

            tam_capa_anterior = len(capa.neuronas)

        # Calculamos las respuestas de la capa de salida
        capa_salida = seriesTemporal.capas[-1]
        valores_capa_salida = np.empty(len(capa_salida.neuronas))
        for j, neurona in enumerate(capa_salida.neuronas):
            valores_capa_salida[j] = neurona.get_valor()
            valores_salida.append(valores_capa_salida[j])

        error_cuadratico = 0
        for j in range(len(valores_capa_salida)):
            resta_cuadrados = clases_test[i][j] - valores_capa_salida[j]
            error_cuadratico += resta_cuadrados ** 2

        ecm_test = error_cuadratico / len(atributos_test)


    # Resultados de la particion de test
    print("ECM test: ", ecm_test)
    print("ECM basico: ", ECM_basico)

    # Prediccion recursiva
    # Lo primero es reintroducir en la red el ultimo dato de entrenamiento y a partir de ese generar los demas
    pred_recursiva_atributos = atributos_train[-1]

    resultados_pred_recursiva = []

    # Hacemos un bucle para que prediga el resto de datos hasta terminar con la misma longitud del dataset.
    for dato in range(len(atributos_test)):

        for j, atributo in enumerate(pred_recursiva_atributos):
            seriesTemporal.neuronas_entrada[j].set_valor(float(atributo))

        ind = 0
        for ind, capa in enumerate(seriesTemporal.capas):
            if ind == 0:
                tam_capa_anterior = len(seriesTemporal.neuronas_entrada)

            seriesTemporal.entrada_neta_capa_a_capa(capa, tam_capa_anterior)
            capa.respuestas_activacion()

            tam_capa_anterior = len(capa.neuronas)

        capa_salida = seriesTemporal.capas[-1]
        valores_capa_salida = np.empty(len(capa_salida.neuronas))
        for j, neurona in enumerate(capa_salida.neuronas):
            valores_capa_salida[j] = neurona.get_valor()
            resultados_pred_recursiva.append(valores_capa_salida[j])

        # Eliminamos el primer elemento y añadimos el que acaba de predecir la serie al final
        pred_recursiva_atributos.pop(0)
        pred_recursiva_atributos.append(*valores_capa_salida)

    # Pintamos las graficas
    # Es necesario modificar el titulo a mano antes de ejecutar
    plt.plot(valores_salida, c= 'red', label = "Valores predichos")
    plt.plot(clases_test,c = 'cyan', label = "Valores test")
    plt.plot(resultados_pred_recursiva, c = '#DCDCDC', label = "Pred. recursiva")

    plt.xlabel("Muestras")
    plt.ylabel("Valores de la serie")
    plt.title("Predicciones serie 2 - Na 5"
              " Ns 1 - 25% train/75% test")
    plt.legend(loc = 'best')
    plt.show()

if __name__ == "__main__":
    main()
