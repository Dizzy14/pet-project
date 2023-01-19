FROM python:3.9

#  Директория для python 3
WORKDIR /usr/src/app

#  Копирование requirements.txt для загрузки всех необходимых библиотек
COPY requirements.txt ./

#  Установка всех необходимых библиотек
RUN pip install --no-cache-dir -r requirements.txt

#  Копирование всех элементов из текущей директории
COPY . .

#  Запуск определенного модуля
CMD [ "python", "bot.py" ]

