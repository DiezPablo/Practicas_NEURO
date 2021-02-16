# Makefile generado para la ejecucion de la practica 2
# Pareja 10 - Pablo Diez y Aljeando Alcala
# Grupo 2462
all:
	@echo "No hace nada, especificar alguna de las siguientes opciones:"
	@echo "  retro_help "
	@echo "  retro_exec2 "
	@echo "  retro_exec6 "
	@echo "Nota: para todos los casos, es necesario tener en un directorio "
	@echo "       los programas y en otro llamado data los ficheros de ejecucion."

retro_help:
	@echo "Para ejecutar correctamente McCullochPitts:"
	@echo "Para ejecutar correctamente el perceptron multicapa:"
	@echo " python3 perceptron.py modo_funcionamiento num_epocas constante_aprendizaje umbral fichero_entrada1 fichero_entrada_test"
	@echo "   - neuronas_capas_ocultas: permite la definicion de neuronas ocultas para la capa de entrada. Ej: 2,3 o 10"
	@echo "   - modo_funcionamiento: modo de funcionamiento del perceptron multicapa, 1, 2 o 3."
	@echo "   - num_epocas: numero de epocas del entrenamient."
	@echo "   - constante_aprendizaje: constante de aprendizaje de la red."
	@echo "   - porcentaje_train: define el porcentaje del fichero que se va a utilizar para entrenar. El resto sera de test."
	@echo "   - normalizacion: si se encuentra a True se realiza normalización sobre los datos. Si está a False no."
	@echo "   - fichero_entrada1: fichero de entrada sobre el que se realiza el entrenamiento"
	@echo "                        y la clasificacion en el caso de los modos de funcionamiento"
	@echo "                        1 y 2. En el modo 3, solo se utiliza para entrenar."
	@echo "   - fichero_entrada_test: este fichero es opcional, solo se utiliza para clasificar"
	@echo "                        cuando el modo de funcionamiento es el 3."

retro_exec2:
	python3 perceptronMulticapa.py 1 10 200 0.1 70% False ../data/problema_real2.txt

retro_exec6:
	@echo "Como comentaste en clase, para 5000 se nos va a ir mucho de tiempo, lo hemos dejado en 250."
	@echo "Aun asi, no se obtienen valores nada buenos."
	python3 perceptronMulticapa.py 1 20 250 0.1 70% True ../data/problema_real6.txt 
