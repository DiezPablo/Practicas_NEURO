import sys
import numpy as np
import random

def main():

    if len(sys.argv) < 5:
        print("Es necesario llamar al programa con los siguientes argumentos:")
        print("   python3 Adapta_fichero_serie.py fichero_entrada fichero_salida Na Ns")
        return

    # Definimos las variables que va a usar el programa
    fichero_entrada = sys.argv[1]
    fichero_salida = sys.argv[2]
    na = int(sys.argv[3]) # Neuronas de entrada
    ns = int(sys.argv[4]) # Neuronas de salida

    fichero_leido = []

    # Leemos el fichero de entrada y lo almacenamos con formato float
    with open (fichero_entrada, "r") as f_in:
        # Leemos el fichero y lo almacenamos eliminando espacios y /n's
        for linea in f_in:
            salida = linea.split("\n")
            salida.remove("")
            fichero_leido.append(*salida)

    fichero_leido = [float(valor) for valor in fichero_leido]

    num_entradas_red_resultante = len(fichero_leido) - na - ns + 1

    # Abrimos el fichero de salida
    with open(fichero_salida, "w") as f_out:

        # Lo primero escribimos el numero de neuronas de entrada y de salida, que coinciden con Na y Ns
        f_out.write(str(na) +" " +str(ns) + "\n")

        # Ahora escribimos las entradas y salidas de la red
        for i in range(num_entradas_red_resultante):
            f_out.write(str(fichero_leido[i]))

            for j in range(na+ns-1):
                f_out.write(" " +str(fichero_leido[i+j+1]))

            f_out.write("\n")



if __name__ == "__main__":
    main()