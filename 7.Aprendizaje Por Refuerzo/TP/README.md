# Guía : Proyecto RL Space Invaders (DQN)
Esta guía permite replicar el entorno de desarrollo utilizado para el proyecto, asegurando compatibilidad con **GPU**, **Gym 0.17** y **TensorFlow 2.12** en Linux/WSL.

## Requisitos Previos

1.  **Sistema Operativo:** Windows con **WSL2** (Ubuntu) o Linux nativo.
2.  **Drivers GPU:** Tener los drivers de NVIDIA actualizados en Windows.
3.  **Software:** Tener instalado **Anaconda** o Miniconda en la terminal de Linux.
4.  **IDE:** Visual Studio Code (con la extensión de Python y Jupyter instalada).

---

## Instalación Paso a Paso

Abre tu terminal de Ubuntu/WSL y ejecuta los siguientes bloques de comandos en orden.

### 1. Preparar el Sistema (Dependencias de compilación)
Necesario para que Gym y Atari se instalen sin errores.
sudo apt-get update
sudo apt-get install -y cmake zlib1g-dev git build-essential
2. Crear el Entorno con Python 3.11
conda create --name 08miar_rl python=3.11 -y
conda activate 08miar_rl
3. ⚡ INSTALACIÓN DE GPU (CUDA + cuDNN)
Instalamos las librerías gráficas directamente en el entorno Conda para evitar conflictos con Windows.

# Instalar librerías NVIDIA compatibles con TF 2.12
conda install -c conda-forge cudatoolkit=11.8.0 cudnn=8.9.2.26 -y

# Configurar variables de entorno para que TensorFlow las encuentre (CRÍTICO)
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
IMPORTANTE: En este punto, cierra tu terminal y vuelve a abrirla, o desactiva y reactiva el entorno para que se cargue la configuración de la GPU: conda deactivate && conda activate 08miar_rl

4. Instalación de Librerías de Python (Versiones Exactas)
Este bloque instala todo lo necesario, solucionando el conflicto de "Numpy" y las "ROMs faltantes".

# Herramientas de compilación
pip install setuptools==65.5.0 wheel

# Gym y Atari (Versión parcheada que incluye ROMs de juegos)
pip install gym==0.17.3
pip install git+[https://github.com/Kojoley/atari-py.git](https://github.com/Kojoley/atari-py.git)

# TensorFlow, Keras-RL y utilidades (Forzando Numpy antiguo para compatibilidad)
pip install tensorflow==2.12.1 keras==2.12.0
pip install keras-rl2==1.0.5
pip install numpy==1.23.5 pandas==1.5.3 matplotlib==3.7.5 pillow openpyxl

# Jupyter y Kernel
pip install ipykernel notebook ipython==8.12.3 typing-extensions==4.5.0
5. Registrar el Kernel en Jupyter
Para que aparezca en VS Code.

python -m ipykernel install --user --name=08miar_rl --display-name "Python (08miar_rl)"

🎮 Cómo ejecutar el proyecto en VS Code
Abre la carpeta del proyecto: code .
Abre el archivo .ipynb.
Arriba a la derecha, haz clic en "Select Kernel" o "Python 3...".
Selecciona la opción: Python Environments -> 08miar_rl.
Ejecuta las celdas.

✅ Verificación de Instalación
Ejecuta la primera celda del notebook. Deberías ver:

GPU Detectada y Configurada ✅

TensorFlow Version: 2.12.1

Gym Version: 0.17.3

🆘 Solución de Problemas Comunes
Error: ImportError: numpy.core.multiarray failed to import: Significa que se ha instalado un Numpy demasiado moderno. Solución: pip install numpy==1.23.5

Error: Could not load dynamic library 'libnvinfer.so.7': Son advertencias normales de TensorRT en WSL. Si dice "GPU Detectada", ignora estos mensajes.

El entrenamiento va lento: Asegúrate de que visualize=False está configurado en la función dqn.fit().

💾 Sistema de Guardado (Checkpoints)
El código está configurado para ser robusto ante fallos:

Guardado: Se crea un checkpoint cada 10.000 pasos en la carpeta /checkpoints.

Recuperación: Al reiniciar el notebook, el código busca automáticamente el archivo más reciente y continúa el entrenamiento desde ahí.