# Pet-project

## SISR

SISR (Single Image Super Resolution) - задача увеличения разрешения одного изображения. Из входного изображения с низким разрешением (Low Resolution, LR) необходимо реконструировать изображение с высоким разрешением (Super Resolution, SR), которое будет максимально похоже на изначальное фото (High Resolution, HR).

## SRGAN

[Оригинальная статья](https://arxiv.org/pdf/1609.04802.pdf)

SRGAN (Super Resolution GAN) - подход к решению задачи SISR, основанный на GAN-ах (генеративно-состязательных сетях).

1) Генеративно-состязательные сети позволяют создавать более реалистичные изображения, чем нейросети, основанные на оптимизации MSE между пикселями. Модели, ориентированные на оптимизацию MSE по пикселям, "усредняли" текстуры, что делало их чрезмерно гладкими. Использование GAN-ов сдвигает реконструированное фото в сторону множества естественных изображений, позволяя получить более реалистичные решения. 

2) Второй важной особенностью стало использование perceptual loss, которая основывается на евклидовых расстояниях (MSE, MAE), вычисленных в пространстве признаков глубокой сверточной нейронной сети (например, предварительно обученной VGG). Такая функция ошибки будет более инвариантна к изменениям пикселей на изображении, чем попиксельные MSE или MAE.

## Архитектура сети

Основу генеративной сети составляют B residual блоков с одинаковой структурой. В каждом блоке находятся два свёрточных слоя с ядрами 3x3 и 64 каналами, за которыми расположены batch-norm слои. В качестве функции активации используется PReLU (Parametric Rectified Linear Unit). Входное изображение увеличивается попиксельно с помощью двух свёрточных слоев с операцией PixelShuffle и функцией активации PReLU.

## Обучение сети

Обучение нейронной сети будет производится на парах изображений LR-HR. Однако не обязательно заранее подготавливать LR и HR пары изображений вместе, достаточно подготовить только High Resolution фотографии. Low Resolution изображения мы сможем получить из HR изображений, уменьшив их билинейной/бикубической интерполяцией прямо во время обучения. Код обучения SRGAN вы можете посмотреть в модуле train_srgan.py.

Обучение модели разделяют на два этапа: сначала на обычной MSE обучают генератор и получают сеть, которую называют SRResNet. Затем обучают SRGAN, проинициализованный весами SRResNet. Такое разделение необходимо, чтобы генератор не выдавал шум на начальных стадиях обучения и дискриминатор не начинал сразу выигрывать. Это позволяет избежать попадания в нежелательный локальный минимум при обучении SRGAN.

## Loss
В обучении генератора используется BCELoss + Perception Loss (который считает MSE между двумя тензорами в пространстве фичей VGG19). Подробнее можно познакомиться в модуле loss.py

## Датасеты на которых обучалась модель:
  1) DIV2K - https://data.vision.ee.ethz.ch/cvl/DIV2K/
  2) Flickr2k - https://drive.google.com/drive/folders/1AAI2a2BmafbeVExLH-l0aZgvPgJCk5Xm
  
## Запуск бота
Для запуска бота запустите модуль bot.py)
