"""
Модуль для предобработки данных PlantVillage датасета
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from collections import Counter
import json


class PlantDiseasePreprocessor:
    """
    Класс для предобработки данных PlantVillage
    """
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Инициализация препроцессора
        
        Args:
            data_dir: путь к директории с данными PlantVillage
            img_size: размер изображений
            batch_size: размер батча
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = []
        self.class_weights = None
        
    def load_data_paths(self):
        """
        Загрузка путей к изображениям и меток классов
        
        Returns:
            image_paths, labels: списки путей и меток
        """
        image_paths = []
        labels = []
        
        print("🔍 Поиск изображений в датасете...")
        
        # Структура: data_dir/Plant___Disease/image.jpg
        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir() and '___' in class_dir.name:
                class_name = class_dir.name
                if class_name not in self.class_names:
                    self.class_names.append(class_name)
                
                for img_file in class_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG']:
                        image_paths.append(str(img_file))
                        labels.append(class_name)
        
        print(f"\n✅ Найдено:")
        print(f"   Изображений: {len(image_paths)}")
        print(f"   Классов: {len(self.class_names)}")
        
        return image_paths, labels
    
    def parse_class_info(self, class_name):
        """
        Извлечение информации о растении и болезни из названия класса
        
        Args:
            class_name: строка вида "Plant___Disease"
            
        Returns:
            dict с информацией
        """
        parts = class_name.split('___')
        if len(parts) == 2:
            plant = parts[0].replace('_', ' ')
            disease = parts[1].replace('_', ' ')
            is_healthy = 'healthy' in disease.lower()
            return {
                'class': class_name,
                'plant': plant,
                'disease': disease,
                'is_healthy': is_healthy
            }
        return {'class': class_name, 'plant': 'Unknown', 'disease': 'Unknown', 'is_healthy': False}
    
    def create_dataframe(self, image_paths, labels):
        """
        Создание DataFrame с дополнительной информацией
        
        Args:
            image_paths: список путей к изображениям
            labels: список меток
            
        Returns:
            df: pandas DataFrame
        """
        print("\n📊 Создание DataFrame...")
        
        # Базовый DataFrame
        df = pd.DataFrame({
            'filepath': image_paths,
            'class': labels
        })
        
        # Добавляем информацию о растениях и болезнях
        class_info = [self.parse_class_info(cls) for cls in df['class']]
        df['plant'] = [info['plant'] for info in class_info]
        df['disease'] = [info['disease'] for info in class_info]
        df['is_healthy'] = [info['is_healthy'] for info in class_info]
        
        # Статистика
        print("\n📈 Статистика по классам:")
        class_counts = df['class'].value_counts()
        print(f"   Всего классов: {len(class_counts)}")
        print(f"   Min samples per class: {class_counts.min()}")
        print(f"   Max samples per class: {class_counts.max()}")
        print(f"   Mean samples per class: {class_counts.mean():.1f}")
        
        print("\n🌱 Растения в датасете:")
        print(df['plant'].value_counts())
        
        print("\n🦠 Здоровые vs Больные:")
        print(df['is_healthy'].value_counts())
        
        return df
    
    def calculate_class_weights(self, df):
        """
        Расчёт весов классов для борьбы с дисбалансом
        
        Args:
            df: DataFrame с данными
            
        Returns:
            class_weights: словарь весов
        """
        class_counts = df['class'].value_counts()
        total = len(df)
        
        # Inverse frequency weighting
        class_weights = {}
        for idx, class_name in enumerate(sorted(class_counts.index)):
            count = class_counts[class_name]
            weight = total / (len(class_counts) * count)
            class_weights[idx] = weight
        
        print(f"\n⚖️  Веса классов рассчитаны")
        print(f"   Min weight: {min(class_weights.values()):.4f}")
        print(f"   Max weight: {max(class_weights.values()):.4f}")
        
        self.class_weights = class_weights
        return class_weights
    
    def split_data(self, df, test_size=0.15, val_size=0.15, random_state=42):
        """
        Стратифицированное разделение данных
        
        Args:
            df: DataFrame с данными
            test_size: доля тестовой выборки
            val_size: доля валидационной выборки
            random_state: seed
            
        Returns:
            train_df, val_df, test_df
        """
        print(f"\n✂️  Разделение данных...")
        
        # Отделяем test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['class']
        )
        
        # Отделяем validation
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_df['class']
        )
        
        print(f"   Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"   Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def create_data_generators(self, train_df, val_df, test_df, 
                               augmentation=True, augment_rare_classes=False):
        """
        Создание генераторов данных
        
        Args:
            train_df, val_df, test_df: DataFrames
            augmentation: применять ли аугментацию
            augment_rare_classes: усиленная аугментация для редких классов
            
        Returns:
            train_generator, val_generator, test_generator
        """
        print(f"\n🔄 Создание генераторов данных...")
        print(f"   Аугментация: {'✓' if augmentation else '✗'}")
        print(f"   Усиленная аугментация редких классов: {'✓' if augment_rare_classes else '✗'}")
        
        # Аугментация для обучающей выборки
        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.15,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # Только нормализация для val/test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Создание генераторов
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='filepath',
            y_col='class',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        val_generator = val_test_datagen.flow_from_dataframe(
            val_df,
            x_col='filepath',
            y_col='class',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_dataframe(
            test_df,
            x_col='filepath',
            y_col='class',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Сохранение маппинга классов
        self.class_indices = train_generator.class_indices
        self.save_class_mapping()
        
        print(f"\n✅ Генераторы созданы")
        print(f"   Классов: {len(train_generator.class_indices)}")
        print(f"   Батчей в train: {len(train_generator)}")
        print(f"   Батчей в val: {len(val_generator)}")
        print(f"   Батчей в test: {len(test_generator)}")
        
        return train_generator, val_generator, test_generator
    
    def save_class_mapping(self, save_path='data/processed/class_mapping.json'):
        """Сохранение маппинга классов"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Создаём расширенный маппинг с информацией
        extended_mapping = {}
        for class_name, idx in self.class_indices.items():
            info = self.parse_class_info(class_name)
            extended_mapping[idx] = {
                'class_name': class_name,
                'plant': info['plant'],
                'disease': info['disease'],
                'is_healthy': info['is_healthy']
            }
        
        with open(save_path, 'w') as f:
            json.dump(extended_mapping, f, indent=4)
        
        print(f"💾 Маппинг классов сохранён: {save_path}")
    
    def prepare_data_pipeline(self, augment_rare_classes=False):
        """
        Полный пайплайн подготовки данных
        
        Returns:
            train_gen, val_gen, test_gen, class_weights
        """
        print("=" * 70)
        print("🌿 ПОДГОТОВКА ДАННЫХ PLANTVILLAGE")
        print("=" * 70)
        
        # 1. Загрузка путей
        image_paths, labels = self.load_data_paths()
        
        # 2. Создание DataFrame
        df = self.create_dataframe(image_paths, labels)
        
        # 3. Расчёт весов классов
        class_weights = self.calculate_class_weights(df)
        
        # 4. Разделение данных
        train_df, val_df, test_df = self.split_data(df)
        
        # 5. Создание генераторов
        train_gen, val_gen, test_gen = self.create_data_generators(
            train_df, val_df, test_df,
            augmentation=True,
            augment_rare_classes=augment_rare_classes
        )
        
        print("\n" + "=" * 70)
        print("✅ ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА")
        print("=" * 70)
        
        return train_gen, val_gen, test_gen, class_weights


def load_and_preprocess_image(image_path, img_size=(224, 224)):
    """
    Загрузка и предобработка одного изображения
    
    Args:
        image_path: путь к изображению
        img_size: размер
        
    Returns:
        preprocessed_image
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    return img_array


if __name__ == "__main__":
    # Пример использования
    preprocessor = PlantDiseasePreprocessor(
        data_dir='data/raw/PlantVillage',
        img_size=(224, 224),
        batch_size=32
    )
    
    train_gen, val_gen, test_gen, class_weights = preprocessor.prepare_data_pipeline()
    
    print("\n✅ Пайплайн готов к обучению!")
