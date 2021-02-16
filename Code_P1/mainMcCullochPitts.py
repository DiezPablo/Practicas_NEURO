from McCullochPitts import *
import sys

def leerDataset(fichero):
    datos = []

    with open(fichero, "r") as file:
        for line in file:
            valores = line[:-1].split(" ")
            datos.append(valores)

    return datos

def main():
	if len(sys.argv) == 1:
		print("Error al introducir los datos de entrada. Falta")
		exit()

	datosEntrada = leerDataset(sys.argv[1])

	# Creacion de neuronas de entrada
	x1 = NeuronaMcCullochPits("x1")
	x2 = NeuronaMcCullochPits("x2")
	x3 = NeuronaMcCullochPits("x3")

	# Creacion de neuronas de salida
	s1 = NeuronaMcCullochPits("s1", 1)
	s2 = NeuronaMcCullochPits("s2", 1)

	z1 = NeuronaMcCullochPits("z1",1)
	z2 = NeuronaMcCullochPits("z2",1)
	z3 = NeuronaMcCullochPits("z3",1)
	y1 = NeuronaMcCullochPits("z4",2)
	y2 = NeuronaMcCullochPits("z5",2)
	y3 = NeuronaMcCullochPits("z6",2)
	y4 = NeuronaMcCullochPits("z7",2)
	y5 = NeuronaMcCullochPits("z8",2)
	y6 = NeuronaMcCullochPits("z9",2)

	# Enlaces entre Entrada y "Primera capa oculta"
	e1 = Enlace(1, x1, z1)
	e2 = Enlace(1, x2, z2)
	e3 = Enlace(1, x3, z3)

	# Enlaces a la "Segunda capa oculta"
	e4 = Enlace(1, x1, y1)
	e5 = Enlace(1, z2, y1)
	e6 = Enlace(1, x2, y2)
	e7 = Enlace(1, z3, y2)
	e8 = Enlace(1, x3, y3)
	e9 = Enlace(1, z1, y3)
	e10 = Enlace(1, x3, y4)
	e11 = Enlace(1, z2, y4)
	e12 = Enlace(1, x2, y5)
	e13 = Enlace(1, z1, y5)
	e14 = Enlace(1, x1, y6)
	e15 = Enlace(1, z3, y6)

	# Enlaces a la capa de salida
	e16 = Enlace(1, y1, s1)
	e17 = Enlace(1, y2, s1)
	e18 = Enlace(1, y3, s1)
	e19 = Enlace(1, y4, s2)
	e20 = Enlace(1, y5, s2)
	e21 = Enlace(1, y6, s2)

	# Crear la RedNeuronal y una capa
	redMccPitts = RedMcCullochPitts("McCullochPitss")

	lista_neuronas = []

	# Añadimos las neuronas a la capa, y la capa a la red neuronal
	lista_neuronas.append(x1)
	lista_neuronas.append(x2)
	lista_neuronas.append(x3)
	lista_neuronas.append(z1)
	lista_neuronas.append(z2)
	lista_neuronas.append(z3)
	lista_neuronas.append(y1)
	lista_neuronas.append(y2)
	lista_neuronas.append(y3)
	lista_neuronas.append(y4)
	lista_neuronas.append(y5)
	lista_neuronas.append(y6)
	lista_neuronas.append(s1)
	lista_neuronas.append(s2)

	# Añadimos los enlaces a la Red Neuronal
	redMccPitts.addEnlace(e1)
	redMccPitts.addEnlace(e2)
	redMccPitts.addEnlace(e3)

	redMccPitts.addEnlace(e4)
	redMccPitts.addEnlace(e5)
	redMccPitts.addEnlace(e6)
	redMccPitts.addEnlace(e7)
	redMccPitts.addEnlace(e8)
	redMccPitts.addEnlace(e9)
	redMccPitts.addEnlace(e10)
	redMccPitts.addEnlace(e11)
	redMccPitts.addEnlace(e12)
	redMccPitts.addEnlace(e13)
	redMccPitts.addEnlace(e14)
	redMccPitts.addEnlace(e15)

	redMccPitts.addEnlace(e16)
	redMccPitts.addEnlace(e17)
	redMccPitts.addEnlace(e18)
	redMccPitts.addEnlace(e19)
	redMccPitts.addEnlace(e20)
	redMccPitts.addEnlace(e21)

	# El numero de instantes de tiempo será el siguiente
	tiempos = len(datosEntrada) +2

	# Creamos el fichero de salida
	f = open("McCulloch_Pitts.out", "w")
	#print(datosEntrada)
	# Simulamos los instantes de tiempo con un bucle for
	for t in range(tiempos):

		print("******************* Instante temporal ",t, "*******************")

		# Escribimos la salida en cada instante de tiempo
		salida_red = str(s1.get_valor()) +" "+str(s2.get_valor()) +"\n"
		f.write(salida_red)

		# Asignamos valore a las neuronas de entrada en cada tiempo t
		if t < len(datosEntrada):
			x1.set_valor(datosEntrada[t][0])
			x2.set_valor(datosEntrada[t][1])
			x3.set_valor(datosEntrada[t][2])
		else:
			x1.set_valor(0)
			x2.set_valor(0)
			x3.set_valor(0)

		print(" - Neuronas de entrada: ",x1.get_valor(), " ", x2.get_valor(), " ",x3.get_valor())
		print(" - Neuronas de salida: ", s1.get_valor(), " ", s2.get_valor())

		# Pasamos los distintos valores por los enlaces hasta las neuronas origen, aplicando las funciones de evaluacion
		# y propagando los valores correspondiente
		for enlace in redMccPitts.enlaces:
			valorNeuronaOrigen = int(enlace.get_neuronaOrigen().get_valor())
			enlace.get_neuronaDestino().set_suma_propagacion(valorNeuronaOrigen)

		# Comprobamos los umbrales para ver las activaciones de las neuronas
		for neurona in lista_neuronas:

			# Comprobamos los umbrales
			neurona.funcion_activacion()

	f.close()



	# Enlaces
if __name__ == "__main__":
	main()
