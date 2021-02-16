import sys
import numpy as np
import math
import re
from matplotlib import pyplot as plt

# Problema real 1 con distintas neuronas en la capa oculta
lista_ECM_evolucion_3_capas = []
lista_ECM_evolucion_2_capas = []
lista_ECM_evolucion_1_capas = []
# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real1.txt_cteLearn_0.1num_capas_3.txt", "r") as f:
#     linea3 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real1.txt_cteLearn_0.1num_capas_2.txt", "r") as f:
#     linea2 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real1.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# for ecm in linea3[:-1]:
#     lista_ECM_evolucion_3_capas.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_ECM_evolucion_2_capas.append(float(ecm))

# for ecm in linea1[:-1]:
#     lista_ECM_evolucion_1_capas.append(float(ecm))
# plt.plot(lista_ECM_evolucion_3_capas, c='blue', label = '3 capas(10,5,2)')
# plt.plot(lista_ECM_evolucion_2_capas, c='green', label = '2 capas(10,5)')
# plt.plot(lista_ECM_evolucionwith open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real1.txt_cteLearn_0.1num_capas_3.txt", "r") as f:
#     linea3 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real1.txt_cteLearn_0.1num_capas_2.txt", "r") as f:
#     linea2 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real1.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# for ecm in linea3[:-1]:
#     lista_ECM_evolucion_3_capas.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_ECM_evolucion_2_capas.append(float(ecm))

# for ecm in linea1[:-1]:
#     lista_ECM_evolucion_1_capas.append(float(ecm))
# plt.plot(lista_ECM_evolucion_3_capas, c='blue', label = '3 capas(10,5,2)')
# plt.plot(lista_ECM_evolucion_2_capas, c='green', label = '2 capas(10,5)')
# plt.plot(lista_ECM_evolucion_1_capas, c='red', label = '1 capa(10)')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 1 - Distinto número de capas oculta - 100 épocas - Cte 0.1")
# plt.legend(loc='best')
# plt.show()

# print("Porcentaje de aciero para 3 capas: ", linea3[-1])
# print("Porcentaje de aciero para 2 capas: ", linea2[-1])
# print("Porcentaje de aciero para 1 capas: ", linea1[-1])_1_capas, c='red', label = '1 capa(10)')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 1 - Distinto número de capas oculta - 100 épocas - Cte 0.1")
# plt.legend(loc='best')
# plt.show()

# print("Porcentaje de aciero para 3 capas: ", linea3[-1])
# print("Porcentaje de aciero para 2 capas: ", linea2[-1])
# print("Porcentaje de aciero para 1 capas: ", linea1[-1])

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real2.txt_cteLearn_0.1num_capas_3.txt", "r") as f:
#     linea3 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real2.txt_cteLearn_0.1num_capas_2.txt", "r") as f:
#     linea2 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real2.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# for ecm in linea3[:-1]:
#     lista_ECM_evolucion_3_capas.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_ECM_evolucion_2_capas.append(float(ecm))

# for ecm in linea1[:-1]:
#     lista_ECM_evolucion_1_capas.append(float(ecm))
# plt.plot(lista_ECM_evolucion_3_capas, c='blue', label = '3 capas(10,5,2)')
# plt.plot(lista_ECM_evolucion_2_capas, c='green', label = '2 capas(10,5)')
# plt.plot(lista_ECM_evolucion_1_capas, c='red', label = '1 capa(10)')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 2 - Distinto número de capas ocultas - 100 épocas -- Cte 0.1")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de aciero para 3 capas: ", linea3[-1])
# print("Porcentaje de aciero para 2 capas: ", linea2[-1])
# print("Porcentaje de aciero para 1 capas: ", linea1[-1])

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real3_clases.txt_cteLearn_0.1num_capas_3.txt", "r") as f:
#     linea3 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real3_clases.txt_cteLearn_0.1num_capas_2.txt", "r") as f:
#     linea2 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real3_clases.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# for ecm in linea3[:-1]:
#     lista_ECM_evolucion_3_capas.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_ECM_evolucion_2_capas.append(float(ecm))

# for ecm in linea1[:-1]:
#     lista_ECM_evolucion_1_capas.append(float(ecm))
# plt.plot(lista_ECM_evolucion_3_capas, c='blue', label = '3 capas(10,5,2)')
# plt.plot(lista_ECM_evolucion_2_capas, c='green', label = '2 capas(10,5)')
# plt.plot(lista_ECM_evolucion_1_capas, c='red', label = '1 capa(10)')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 3 - Distinto número de capas ocultas - 100 épocas -- Cte 0.1")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de aciero para 3 capas: ", linea3[-1])
# print("Porcentaje de aciero para 2 capas: ", linea2[-1])
# print("Porcentaje de aciero para 1 capas: ", linea1[-1])
# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real5.txt_cteLearn_0.1num_capas_3.txt", "r") as f:
#     linea3 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real5.txt_cteLearn_0.1num_capas_2.txt", "r") as f:
#     linea2 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real5.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# for ecm in linea3[:-1]:
#     lista_ECM_evolucion_3_capas.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_ECM_evolucion_2_capas.append(float(ecm))

# for ecm in linea1[:-1]:
#     lista_ECM_evolucion_1_capas.append(float(ecm))
# plt.plot(lista_ECM_evolucion_3_capas, c='blue', label = '3 capas(10,5,2)')
# plt.plot(lista_ECM_evolucion_2_capas, c='green', label = '2 capas(10,5)')
# plt.plot(lista_ECM_evolucion_1_capas, c='red', label = '2 capas(10)')
# plt.title("ECM - Problema 3 - Distinto número de capas ocultas - 100 épocas -- Cte 0.1")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de aciero para 3 capas: ", linea3[-1])
# print("Porcentaje de aciero para 2 capas: ", linea2[-1])
# print("Porcentaje de aciero para 1 capas: ", linea1[-1])
# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real2.txt_cteLearn_0.1num_capas_3.txt", "r") as f:
#     linea3 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real2.txt_cteLearn_0.1num_capas_2.txt", "r") as f:
#     linea2 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real2.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# for ecm in linea4[:-1]:
#     lista_ECM_evolucion_3_capas.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_ECM_evolucion_2_capas.append(float(ecm))

# for ecm in linea1[:-1]:
#     lista_ECM_evolucion_1_capas.append(float(ecm))
# plt.plot(lista_ECM_evolucion_3_capas, c='blue', label = '3 capas(10,5,2)')
# plt.plot(lista_ECM_evolucion_2_capas, c='green', label = '2 capas(10,5)')
# plt.plot(lista_ECM_evolucion_1_capas, c='red', label = '1 capa(10)')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 2 - Distinto numero de capa oculta")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de aciero para 3 capas: ", linea3[-1])
# print("Porcentaje de aciero para 2 capas: ", linea2[-1])
# print("Porcentaje de aciero para 1 capas: ", linea1[-1])
# plt.plot(lista_ECM_evolucion_1_capas, c='red', label = '1 capa(10)')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 5 - Distinto numero de capa oculta")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de aciero para 3 capas: ", linea3[-1])
# print("Porcentaje de aciero para 2 capas: ", linea2[-1])
# print("Porcentaje de aciero para 1 capas: ", linea1[-1])


# Comparacion entre distintas constantes de aprendizaje con una capa
# Comparacion, para mismo numero de epocas y distintas constantes de aprendizaje
lista_cte_01_100 = []
lista_cte_0001_100 = []

lista_cte_01_200 = []
lista_cte_0001_200 = []

lista_cte_01_500 = []
lista_cte_0001_500 = []
# Para el problema real 1

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real1.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real1.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_cte_01_100.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_cte_0001_100.append(float(ecm))

# plt.plot(lista_cte_01_100, c='blue', label = 'k-Learn 0.1')
# plt.plot(lista_cte_0001_100, c='green', label = 'k-Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 1 - Variacion k-Learn - 100 épocas - 10 neu. ocultas")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 100 epocas - k-Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 100 epocas - k-Learn 0.001: ", linea2[-1])

# with open("Resultados_Pruebas/ECM_n_epocas_200 _problema_real1.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_200 _problema_real1.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea3[:-1]:
#     lista_cte_01_200.append(float(ecm))

# for ecm in linea4[:-1]:
#     lista_cte_0001_200.append(float(ecm))

# plt.plot(lista_cte_01_200, c='blue', label = '200 epocas - Cte Learn 0.1')
# plt.plot(lista_cte_0001_200, c='green', label = '200 epocas - Cte Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 1 - Variacion de constante de aprendizaje")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 200 epocas - Cte Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 200 epocas - Cte Learn 0.001: ", linea2[-1])

# with open("Resultados_Pruebas/ECM_n_epocas_500 _problema_real1.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_500 _problema_real1.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_cte_01_500.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_cte_0001_500.append(float(ecm))

# plt.plot(lista_cte_01_500, c='blue', label = 'k-Learn 0.1')
# plt.plot(lista_cte_0001_500, c='green', label = 'k-Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 1 - Variacion k-Learn - 500 épocas - 10 neu. ocultas")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 500 epocas - Cte Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 500 epocas - Cte Learn 0.001: ", linea2[-1])

# Comparacion constantes de aprendizaje Problema_real2
# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real2.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real2.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_cte_01_100.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_cte_0001_100.append(float(ecm))

# plt.plot(lista_cte_01_100, c='blue', label = 'k-Learn 0.1')
# plt.plot(lista_cte_0001_100, c='green', label = 'k-Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 2 - Variacion k-Learn - 500 épocas - 10 neu. ocultas")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 100 epocas - Cte Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 100 epocas - Cte Learn 0.001: ", linea2[-1])

# with open("Resultados_Pruebas/ECM_n_epocas_200 _problema_real2.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_200 _problema_real2.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_cte_01_200.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_cte_0001_200.append(float(ecm))

# plt.plot(lista_cte_01_200, c='blue', label = 'k-Learn 0.1')
# plt.plot(lista_cte_0001_200, c='green', label = 'k-Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 2 - Variacion k-Learn - 500 épocas - 10 neu. ocultas")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 200 epocas - Cte Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 200 epocas - Cte Learn 0.001: ", linea2[-1])

# with open("Resultados_Pruebas/ECM_n_epocas_500 _problema_real2.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_500 _problema_real2.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_cte_01_500.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_cte_0001_500.append(float(ecm))

# plt.plot(lista_cte_01_500, c='blue', label = 'k-Learn 0.1')
# plt.plot(lista_cte_0001_500, c='green', label = 'k-Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 2 - Variacion k-Learn - 500 épocas - 10 neu. ocultas")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 500 epocas - Cte Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 500 epocas - Cte Learn 0.001: ", linea2[-1])

# Problema real3_clases

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real3_clases.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real3_clases.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_cte_01_100.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_cte_0001_100.append(float(ecm))

# plt.plot(lista_cte_01_100, c='blue', label = 'k-Learn 0.1')
# plt.plot(lista_cte_0001_100, c='green', label = 'k-Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 3 - Variacion k-Learn - 100 épocas - 10 neu. ocultas")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 100 epocas - k-Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 100 epocas - k-Learn 0.001: ", linea2[-1])

# with open("Resultados_Pruebas/ECM_n_epocas_200 _problema_real3_clases.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_200 _problema_real3_clases.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea3[:-1]:
#     lista_cte_01_200.append(float(ecm))

# for ecm in linea4[:-1]:
#     lista_cte_0001_200.append(float(ecm))

# plt.plot(lista_cte_01_200, c='blue', label = '200 epocas - Cte Learn 0.1')
# plt.plot(lista_cte_0001_200, c='green', label = '200 epocas - Cte Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 2 - Variacion de constante de aprendizaje")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 200 epocas - Cte Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 200 epocas - Cte Learn 0.001: ", linea2[-1])

# with open("Resultados_Pruebas/ECM_n_epocas_500 _problema_real3_clases.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_500 _problema_real3_clases.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_cte_01_500.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_cte_0001_500.append(float(ecm))

# plt.plot(lista_cte_01_500, c='blue', label = 'k-Learn 0.1')
# plt.plot(lista_cte_0001_500, c='green', label = 'k-Cte Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 3 - Variacion k-Learn - 500 épocas - 10 neu. ocultas")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 500 epocas - k-Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 500 epocas - k-Learn 0.001: ", linea2[-1])

# Problema real 5

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real5.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real5.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_cte_01_100.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_cte_0001_100.append(float(ecm))

# plt.plot(lista_cte_01_100, c='blue', label = 'k-Learn 0.1')
# plt.plot(lista_cte_0001_100, c='green', label = 'k-Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 5 - Variacion k-Learn - 100 épocas - 10 neu. ocultas")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 100 epocas - Cte Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 100 epocas - Cte Learn 0.001: ", linea2[-1])

# with open("Resultados_Pruebas/ECM_n_epocas_200 _problema_real5.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_200 _problema_real5.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea3[:-1]:
#     lista_cte_01_200.append(float(ecm))

# for ecm in linea4[:-1]:
#     lista_cte_0001_200.append(float(ecm))

# plt.plot(lista_cte_01_200, c='blue', label = '200 epocas - Cte Learn 0.1')
# plt.plot(lista_cte_0001_200, c='green', label = '200 epocas - Cte Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 5 - Variacion de constante de aprendizaje")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 200 epocas - k-Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 200 epocas - k-Learn 0.001: ", linea2[-1])


# with open("Resultados_Pruebas/ECM_n_epocas_500 _problema_real5.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_500 _problema_real5.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_cte_01_500.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_cte_0001_500.append(float(ecm))

# plt.plot(lista_cte_01_500, c='black', label = 'k-Learn 0.1')
# plt.plot(lista_cte_0001_500, c='cyan', label = 'k-Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 5 - Variacion k-Learn - 500 épocas - 10 neu. ocultas")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 500 epocas - Cte Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 500 epocas - Cte Learn 0.001: ", linea2[-1])



############################## GRAFICAS NORMALIZACION###############################################
# Problema real 4
# with open("Resultados_Pruebas/ECM_n_epocas_500 _problema_real4.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_500 _problema_real4.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_cte_01_500.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_cte_0001_500.append(float(ecm))

# plt.plot(lista_cte_01_500, c='blue', label = 'k-Learn 0.1')
# plt.plot(lista_cte_0001_500, c='green', label = 'k-Cte Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 4 - Variacion k-Learn - 500 épocas - 10 neu. ocultas")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 500 epocas - Cte Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 500 epocas - Cte Learn 0.001: ", linea2[-1])


# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real4.txt_cteLearn_0.1num_capas_1.txt", "r") as f:
#     linea1 = f.readline().split(",")

# with open("Resultados_Pruebas/ECM_n_epocas_100 _problema_real4.txt_cteLearn_0.001num_capas_1.txt", "r") as f:
#     linea2 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_cte_01_100.append(float(ecm))

# for ecm in linea2[:-1]:
#     lista_cte_0001_100.append(float(ecm))

# plt.plot(lista_cte_01_100, c='blue', label = 'k-Learn 0.1')
# plt.plot(lista_cte_0001_100, c='green', label = 'k-Learn 0.001')
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 4 - Variacion k-Learn - 500 épocas - 10 neu. ocultas")
# plt.legend(loc='best')
# plt.show()
# print("Porcentaje de acierto 100 epocas - Cte Learn 0.1: ", linea1[-1])
# print("Porcentaje de acierto 100 epocas - Cte Learn 0.001: ", linea2[-1])

#### graficas ejercicio 4 y 6
lista_ecm_ej4_sin_norm = []
lista_ecm_ej4_con_norm = []
lista_ecm_ej6_sin_norm = []
lista_ecm_ej6_con_norm = []

# Evolucion ECM 4 sin norm
# with open ("ECM_n_epocas_100 _problema_real4.txt_cteLearn_0.1num_capas_1_False.txt","r") as f:
#     linea1 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_ecm_ej4_sin_norm.append(float(ecm))

# plt.plot(lista_ecm_ej4_sin_norm)
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 4 - kLearn 0.1 - 100 epocas - 10 neur. oculta - Sin normalizar")
# plt.show()
# print("porc acierto:",str(linea1[-1]))

# with open ("ECM_n_epocas_100 _problema_real4.txt_cteLearn_0.1num_capas_1_True.txt","r") as f:
#     linea1 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_ecm_ej4_con_norm.append(float(ecm))

# plt.plot(lista_ecm_ej4_con_norm)
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 4 - kLearn 0.1 - 100 epocas - 10 neur. oculta- Normalizado")
# plt.show()
# print("porc acierto:",str(linea1[-1]))

# Evolucion ECM 4 sin norm
# with open ("ECM_n_epocas_100 _problema_real6.txt_cteLearn_0.1num_capas_1_False.txt","r") as f:
#     linea1 = f.readline().split(",")

# for ecm in linea1[:-1]:
#     lista_ecm_ej6_sin_norm.append(float(ecm))

# plt.plot(lista_ecm_ej6_sin_norm)
# plt.xlabel("Epocas")
# plt.ylabel("Error cuadratico medio")
# plt.title("ECM - Problema 6 - kLearn 0.1 - 100 epocas - 10 neur. oculta - Sin normalizar")
# plt.show()
# print("porc acierto:",str(linea1[-1]))

with open ("ECM_n_epocas_100 _problema_real6.txt_cteLearn_0.1num_capas_1_True.txt","r") as f:
    linea1 = f.readline().split(",")

for ecm in linea1[:-1]:
    lista_ecm_ej6_con_norm.append(float(ecm))

plt.plot(lista_ecm_ej6_con_norm)
plt.xlabel("Epocas")
plt.ylabel("Error cuadratico medio")
plt.title("ECM - Problema 4 - kLearn 0.1 - 100 epocas - 10 neur. oculta - Normalizado")
plt.show()
print("porc acierto:",str(linea1[-1]))