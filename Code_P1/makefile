# Makefile generado para la ejecucion de la practica 1
# Pareja 10 - Pablo Diez y Aljeando Alcala
# Grupo 2462

all:
	@echo "No hace nada, especificar alguna de las siguientes opciones:"
	@echo "  mp_help "
	@echo "  mp_exec "
	@echo "  per_help "
	@echo "  per_exec "
	@echo "  per_exec_modo2"
	@echo "  per_exec_modo3"
	@echo "  per_exec_nand"
	@echo "  per_exec_nor"
	@echo "  per_exec_xor"
	@echo "  ada_help "
	@echo "  ada_exec "
	@echo "  ada_exec_modo2"
	@echo "  ada_exec_modo3"
	@echo "  ada_exec_nand"
	@echo "  ada_exec_nor"
	@echo "  ada_exec_xor"
	@echo "Nota: para todos los casos, es necesario tener en un directorio "
	@echo "       los programas y en otro llamado data los ficheros de ejecucion."

mp_help:
	@echo "Para ejecutar correctamente McCullochPitts:"
	@echo "  python3 mainMcCullochPitts.py fichero_entrada"
	@echo "   - fichero_entrada: fichero con la entrada que va a evaluar la red."

mp_exec:
	python3 mainMcCullochPitts.py "../data/McCulloch_Pitts.txt"

per_help:
	@echo "Para ejecutar correctamente el perceptron:"
	@echo " python3 perceptron.py modo_funcionamiento num_epocas constante_aprendizaje umbral fichero_entrada1 fichero_entrada_test"
	@echo "   - modo_funcionamiento: modo de funcionamiento del perceptron, 1, 2 o 3."
	@echo "   - num_epocas: numero de epocas del entrenamient."
	@echo "   - constante_aprendizaje: constante de aprendizaje de la red."
	@echo "   - umbral: umbral de activacion de la neurona de salida de perceptron."
	@echo "   - fichero_entrada1: fichero de entrada sobre el que se realiza el entrenamiento"
	@echo "                        y la clasificacion en el caso de los modos de funcionamiento"
	@echo "                        1 y 2. En el modo 3, solo se utiliza para entrenar."
	@echo "   - fichero_entrada_test: este fichero es opcional, solo se utiliza para clasificar"
	@echo "                        cuando el modo de funcionamiento es el 3."

per_exec:
	python3 perceptron.py 1 500 1 0.0 ../data/problema_real2.txt
per_exec_modo2:
	python3 perceptron.py 2 500 1 0.0 ../data/problema_real2.txt
per_exec_modo3:
	python3 perceptron.py 3 500 1 0.0 ../data/problema_real2.txt ../data/problema_real2_no_etiquetados.txt
per_exec_nand:
	python3 perceptron.py 2 100 1 0.2 ../data/nand.txt
per_exec_nor:
	python3 perceptron.py 2 100 1 0.2 ../data/nor.txt
per_exec_xor:
	python3 perceptron.py 2 100 1 0.2 ../data/xor.txt

ada_help:
		@echo "Para ejecutar correctamente el adaline:"
		@echo " python3 adaline.py modo_funcionamiento num_epocas constante_aprendizaje fichero_entrada1 fichero_entrada_test"
		@echo "   - modo_funcionamiento: modo de funcionamiento del adaline, 1, 2 o 3."
		@echo "   - num_epocas: numero de epocas del entrenamient."
		@echo "   - constante_aprendizaje: constante de aprendizaje de la red."
		@echo "   - fichero_entrada1: fichero de entrada sobre el que se realiza el entrenamiento"
		@echo "                        y la clasificacion en el caso de los modos de funcionamiento"
		@echo "                        1 y 2. En el modo 3, solo se utiliza para entrenar."
		@echo "   - fichero_entrada_test: este fichero es opcional, solo se utiliza para clasificar"
		@echo "                        cuando el modo de funcionamiento es el 3."

ada_exec:
	python3 adaline.py 1 100 0.1 ../data/problema_real2.txt
ada_exec_modo2:
	python3 adaline.py 2 100 0.1 ../data/problema_real2.txt
ada_exec_modo3:
	python3 adaline.py 3 100 0.1 ../data/problema_real2.txt ../data/problema_real2_no_etiquetados.txt
ada_exec_nand:
	python3 adaline.py 2 100 0.1 ../data/nand.txt
ada_exec_nor:
	python3 adaline.py 2 100 0.1 ../data/nor.txt
ada_exec_xor:
	python3 adaline.py 2 100 0.1 ../data/xor.txt
