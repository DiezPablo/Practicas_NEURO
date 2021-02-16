import sys
import numpy as np
import random


def main():

    # Comprobacion de errores
    if len(sys.argv) < 5:
        print("Es necesario ejecutar el programa con los siguientes parametros:")
        print("  - python3 Crear_alfabeto num_copias num_errores fich_entrada fich_salida")
        return

    # Almacenamos los parametros de entrada del programa
    num_copias = int(sys.argv[1])
    num_errores = int(sys.argv[2])
    fichero_entrda = sys.argv[3]
    fichero_salida = sys.argv[4]

    # Lista con todas las letras del abecedario

    # Comprobacion de errores
    if (num_copias > 0) and (num_errores <= 0):
        print("El parametro num_copias solo puede ser > 0 cuando num_errores > 0")

    # Almacena como listas la informacion del fichero formateada
    fichero_leido = []

    # Leemos el fichero de entrada
    with open (fichero_entrda, "r") as f_in:
        # Leemos el fichero y lo almacenamos eliminando espacios y /n's
        for linea in f_in:
            salida = linea.split("\n")
            salida.remove("")
            fichero_leido.append(salida)

    fichero_leido = [lista[0].split("     ") for lista in fichero_leido if lista != ['']]

    # Generamos un diccionario con todas las letras del abecedario como clave y su valor como cadena de binarios
    letras_dict = {}
    for i, dato in enumerate(fichero_leido):
        if("Letra" in dato[0]):
            letra = dato[0].split(" ")
            letra = letra[2].split(":")
            letra_valor = []
        else:
             letra_valor += dato
        if i != 0:
            letras_dict[letra[0]] = letra_valor

    # for value in letras_dict.values():
    #     for j, atrib in enumerate(value):
    #         if atrib == '0':
    #             value[j] = '-1'

    # Generacion del fichero de salida en el caso de que num_copias sea <0
    with open(fichero_salida, "w") as f_out:

        # Escritura de la primera linea de todos los ficheros, n_entradas, n_salidas
        f_out.write(str(len(list(letras_dict.values())[0])) +" " +str(len(list(letras_dict.values())[0])) +"\n")
        for letra in letras_dict:
            if num_copias != 0:
                for i in range(num_copias):

                    # Copiamos la letra original para no modificarla entre iteraciones
                    letra_final = letras_dict[letra].copy()

                    # Para cada copia generamos una lista de num_errores con los indices entre el 0 y el 34
                    indices_errores = random.sample(range(35), num_errores)

                    # Modificamos esos valores
                    for indice in indices_errores:
                        if letra_final[indice] == '1':
                            letra_final[indice] = '0'
                        else:
                            letra_final[indice] = '1'

                    # Escribimos en el fichero de salida los atributos
                    for pixel in letra_final:
                        f_out.write(str(pixel) +" ")

                    # Escribimos en el fichero de salida las clases de salida
                    for i, pixel in enumerate(letras_dict[letra]):
                        if i != (len(list(letras_dict[letra])) - 1):
                            f_out.write((str(pixel)) +" ")
                        else:
                            f_out.write((str(pixel)))
                    f_out.write("\n")
            else:

                # Escribimos en el fichero de salida los atributos
                for pixel in letras_dict[letra]:
                    f_out.write((str(pixel)) + " ")

                # Escribimos en el fichero de salida las clases
                for i, pixel in enumerate(letras_dict[letra]):
                    if i != (len(list(letras_dict[letra])) - 1):
                        f_out.write((str(pixel)) + " ")
                    else:
                        f_out.write((str(pixel)))
                f_out.write("\n")

if __name__== "__main__":
    main()