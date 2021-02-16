# Makefile generado para la ejecucion de la practica 3
# Pareja 10 - Pablo Diez y Aljeando Alcala
# Grupo 2462
all:
	@echo "No hace nada, especificar alguna de las siguientes opciones:"
	@echo "  autoencoder_help "
	@echo "  autoencoder_exec "
	@echo "  serie_help "
	@echo "  serie_exec "
	@echo "Nota: para todos los casos, es necesario tener los directorios de la "
	@echo "       misma manera que figuran en la entrega."

autoencoder_help:
	@echo "Para ejecutar correctamente el autoencoder:"
	@echo " python3 Autoencoder.py neuronas_capas_ocultas num_epocas constante_aprendizaje fichero_entrada_train fichero_entrada_test"
	@echo "   - neuronas_capas_ocultas: permite la definicion de neuronas ocultas para la capa de entrada. Ej: 2,3 o 10"
	@echo "   - num_epocas: numero de epocas del entrenamiento."
	@echo "   - constante_aprendizaje: constante de aprendizaje de la red."
	@echo "   - fichero_entrada_train: fichero utilizado para hacer el train."
	@echo "   - fichero_entrada_test: fichero utilizado para hacer el test."

autoencoder_exec:
	python3 Autoencoder.py 40 200 0.01 Ficheros_alfabeto/salida_completa00.txt Ficheros_alfabeto/salida_completa00.txt

serie_help:
	@echo "Para ejecutar correctamente la serie temporal:"
	@echo " python3 seriesTemporales.py modo_ejecucion neuronas_capas_ocultas num_epocas constante_aprendizaje porcentaje_train fichero_entrada"
	@echo "   - modo_funcionamiento: modo de funcionamiento del predictor, 1, 2 o 3. Unicamente se usa el 1"
	@echo "   - neuronas_capas_ocultas: permite la definicion de neuronas ocultas para la capa de entrada. Ej: 2,3 o 10"
	@echo "   - num_epocas: numero de epocas del entrenamiento."
	@echo "   - constante_aprendizaje: constante de aprendizaje de la red."
	@echo "   - porcentaje_train: porcentaje que se va a usar del fichero de entrada para train, el resto sera test."
	@echo "   - fichero_entrada_train: fichero utilizado para hacer el train."
	


serie_exec:
	python3 seriesTemporales.py 1 10 200 0.01 25% Ficheros_series_temporales/serie1_21.txt
	

