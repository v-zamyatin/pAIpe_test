# Базовый образ с поддержкой TensorFlow и CUDA
FROM tensorflow/tensorflow:2.15.0-gpu

# Установка системных пакетов
RUN apt-get update && apt-get install -y \
    git wget curl unzip nano htop build-essential \
    tabix bedtools gawk coreutils \
    && apt-get clean

# Установка Python-библиотек
RUN pip install --upgrade pip \
    && pip install \
        h5py \
        pandas \
        numpy \
        joblib \
        scikit-learn

# 1. Клонируем borzoi
WORKDIR /opt
RUN git clone https://github.com/calico/borzoi.git
WORKDIR /opt/borzoi

# Устанавливаем borzoi
RUN pip install -e .

# Патчим borzoi_sad.py: добавляем mixed precision + JIT на строку 34
RUN sed -i '34i from tensorflow.keras import mixed_precision\nmixed_precision.set_global_policy("mixed_float16")\ntf.config.optimizer.set_jit(True)\n' /opt/borzoi/src/scripts/borzoi_sad.py

# 2. Клонируем и ставим baskerville
WORKDIR /opt
RUN git clone https://github.com/calico/baskerville.git
WORKDIR /opt/baskerville
RUN pip install -e .

# 3. Назад в borzoi, скачиваем модели
WORKDIR /opt/borzoi
RUN ./download_models.sh

# 4. Переменные окружения для Borzoi и Baskerville
ENV BORZOI_DIR=/opt/borzoi
ENV BASKERVILLE_DIR=/opt/baskerville
ENV BORZOI_CONDA=/opt/conda/etc/profile.d/conda.sh
ENV BASKERVILLE_CONDA=$BORZOI_CONDA
ENV BORZOI_HG38=$BORZOI_DIR/examples/hg38
ENV BORZOI_MM10=$BORZOI_DIR/examples/mm10

# Обновляем PATH и PYTHONPATH для borzoi и baskerville
ENV PATH=$BORZOI_DIR/src/scripts:$BASKERVILLE_DIR/src/baskerville/scripts:$PATH
ENV PYTHONPATH=$BORZOI_DIR/src/scripts:$BASKERVILLE_DIR/src/baskerville/scripts
#:$PYTHONPATH

# 5. Bash aliases для удобства запуска
RUN echo "\n# Borzoi + Baskerville aliases" >> ~/.bashrc && \
    echo "alias borzoi_sad='python $BORZOI_DIR/src/scripts/borzoi_sad.py'" >> ~/.bashrc && \
    echo "alias borzoi_sed='python $BORZOI_DIR/src/scripts/borzoi_sed.py'" >> ~/.bashrc && \
    echo "alias borzoi_predict='python $BORZOI_DIR/src/scripts/predict.py'" >> ~/.bashrc && \
    echo "alias baskerville_build='python $BASKERVILLE_DIR/src/baskerville/scripts/build_dataset.py'" >> ~/.bashrc

# 6. Установка AlphaGenome
RUN pip install alphagenome openpyxl
# 7. Рабочая папка
WORKDIR /workspace
VOLUME ["/workspace"]

ENTRYPOINT ["/bin/bash"]
