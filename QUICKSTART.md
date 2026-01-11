# 🚀 Быстрый старт - Классификация болезней растений

## 📥 Получение данных

### Вариант 1: GitHub (рекомендуется)

```bash
cd data/raw/
git clone https://github.com/spMohanty/PlantVillage-Dataset.git PlantVillage
cd ../..
```

### Вариант 2: Kaggle

```bash
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d data/raw/PlantVillage/
```

## ⚡ Быстрое обучение

```bash
# Установка зависимостей
pip install -r requirements.txt

# Обучение модели (30 эпох)
python src/train.py --epochs 30 --batch_size 32

# Оценка
python src/evaluate.py

# Предсказание
python src/predict.py --image_path path/to/leaf.jpg --show_image
```

## 🎯 Параметры обучения

```bash
# С использованием весов классов (для несбалансированных данных)
python src/train.py --use_class_weights --epochs 30

# С fine-tuning
python src/train.py --fine_tune --fine_tune_epochs 15 --epochs 30

# Другая модель
python src/train.py --model_type resnet50 --epochs 30

# С TensorBoard
python src/train.py --use_tensorboard --epochs 30
# Запустите: tensorboard --logdir=logs
```

## 📊 Ожидаемые результаты

- **Accuracy**: 95-98%
- **Top-3 Accuracy**: 99%+
- **Время обучения**: 
  - С GPU: 30-60 минут (30 эпох)
  - С CPU: 3-5 часов

## 🏥 Использование модели

### Диагностика болезни по фото

```python
import tensorflow as tf
import json
from PIL import Image
import numpy as np

# Загрузка модели
model = tf.keras.models.load_model('models/best_model.h5')

# Загрузка конфигурации
with open('models/config.json', 'r') as f:
    config = json.load(f)
class_names = config['class_names']

# Предсказание
img = Image.open('leaf.jpg').resize((224, 224))
img_array = np.array(img) / 255.0
img_batch = np.expand_dims(img_array, 0)

predictions = model.predict(img_batch)
top_class = class_names[np.argmax(predictions)]

print(f"Диагноз: {top_class}")
```

## 📚 Структура классов PlantVillage

38 классов, включая:

**Яблоня:**
- Apple___Apple_scab
- Apple___Black_rot
- Apple___Cedar_apple_rust
- Apple___healthy

**Томат:**
- Tomato___Bacterial_spot
- Tomato___Early_blight
- Tomato___Late_blight
- Tomato___Leaf_Mold
- Tomato___healthy
- и другие...

**Картофель, виноград, кукуруза** и другие культуры

## 🔍 Анализ результатов

Все результаты сохраняются в `reports/`:
- `confusion_matrix.png` - матрица ошибок
- `training_history.png` - графики обучения
- `predictions_sample.png` - примеры предсказаний
- `classification_report.txt` - детальный отчёт

## 💡 Советы

1. **Несбалансированные классы?** Используйте `--use_class_weights`
2. **Переобучение?** Уменьшите epochs или добавьте Dropout
3. **Низкая точность?** Попробуйте `--fine_tune`
4. **Медленное обучение?** Уменьшите `--batch_size` или используйте MobileNetV2

## 🐛 Решение проблем

**Ошибка: "Dataset not found"**
```bash
# Проверьте структуру
ls data/raw/PlantVillage/
# Должны быть папки типа: Apple___healthy, Tomato___Bacterial_spot, etc.
```

**Ошибка памяти (OOM)**
```bash
python src/train.py --batch_size 16  # Уменьшите batch
```

**Долгое обучение**
```bash
python src/train.py --model_type mobilenetv2  # Более лёгкая модель
```
