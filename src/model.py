"""
Модуль с архитектурой модели для классификации болезней растений
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    EfficientNetB3, ResNet50, DenseNet121, MobileNetV2
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import os
from datetime import datetime


class PlantDiseaseModel:
    """
    Класс для создания и обучения модели классификации болезней растений
    """
    
    def __init__(self, num_classes, img_size=(224, 224), model_type='efficientnetb3'):
        """
        Инициализация
        
        Args:
            num_classes: количество классов
            img_size: размер входных изображений
            model_type: тип базовой модели
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_type = model_type
        self.model = None
        
    def build_model(self, trainable_base=False):
        """
        Построение модели
        
        Args:
            trainable_base: делать ли базовую модель trainable
            
        Returns:
            model: скомпилированная модель
        """
        input_shape = (*self.img_size, 3)
        
        # Выбор базовой модели
        if self.model_type == 'efficientnetb3':
            base_model = EfficientNetB3(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif self.model_type == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif self.model_type == 'densenet121':
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif self.model_type == 'mobilenetv2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
        
        # Заморозка базовой модели
        base_model.trainable = trainable_base
        
        # Построение полной модели
        inputs = keras.Input(shape=input_shape)
        
        # Базовая модель
        x = base_model(inputs, training=False)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Дополнительные Dense слои
        x = layers.Dense(256, activation='relu', name='dense_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(128, activation='relu', name='dense_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax',
            name='output'
        )(x)
        
        # Создание модели
        model = keras.Model(inputs=inputs, outputs=outputs, name=f'{self.model_type}_plant_disease')
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.0001, class_weights=None):
        """
        Компиляция модели
        
        Args:
            learning_rate: learning rate
            class_weights: веса классов для борьбы с дисбалансом
        """
        if self.model is None:
            raise ValueError("Сначала создайте модель с build_model()")
        
        # Метрики
        metrics = [
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=metrics
        )
        
        self.class_weights = class_weights
        print("✅ Модель скомпилирована")
        
    def get_callbacks(self, checkpoint_path='models/best_model.h5', 
                     use_tensorboard=False):
        """
        Создание callbacks
        
        Args:
            checkpoint_path: путь для сохранения модели
            use_tensorboard: использовать ли TensorBoard
            
        Returns:
            список callbacks
        """
        callbacks = [
            # Early Stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model Checkpoint
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce Learning Rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # TensorBoard (опционально)
        if use_tensorboard:
            log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
            callbacks.append(
                TensorBoard(log_dir=log_dir, histogram_freq=1)
            )
        
        return callbacks
    
    def train(self, train_generator, val_generator, epochs=30, 
             callbacks=None, use_class_weights=True):
        """
        Обучение модели
        
        Args:
            train_generator: генератор обучающих данных
            val_generator: генератор валидационных данных
            epochs: количество эпох
            callbacks: список callbacks
            use_class_weights: использовать ли веса классов
            
        Returns:
            history: история обучения
        """
        if self.model is None:
            raise ValueError("Сначала создайте и скомпилируйте модель")
        
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        print(f"\n🚀 Начало обучения модели {self.model_type}")
        print(f"   Эпохи: {epochs}")
        print(f"   Классов: {self.num_classes}")
        print(f"   Train batches: {len(train_generator)}")
        print(f"   Val batches: {len(val_generator)}")
        print(f"   Веса классов: {'✓' if use_class_weights and self.class_weights else '✗'}")
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=self.class_weights if use_class_weights else None,
            verbose=1
        )
        
        print("\n✅ Обучение завершено!")
        
        return history
    
    def fine_tune(self, train_generator, val_generator, epochs=15,
                 unfreeze_layers=50, learning_rate=1e-5):
        """
        Fine-tuning модели
        
        Args:
            train_generator: генератор обучающих данных
            val_generator: генератор валидационных данных
            epochs: количество эпох
            unfreeze_layers: сколько слоёв разморозить
            learning_rate: learning rate
            
        Returns:
            history: история обучения
        """
        print(f"\n🔧 Fine-tuning: размораживаем последние {unfreeze_layers} слоёв")
        
        # Получаем базовую модель
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Замораживаем все кроме последних N слоёв
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Перекомпиляция с меньшим learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Обучение
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=self.get_callbacks(checkpoint_path='models/finetuned_model.h5'),
            class_weight=self.class_weights,
            verbose=1
        )
        
        print("\n✅ Fine-tuning завершён!")
        
        return history
    
    def summary(self):
        """Вывод архитектуры модели"""
        if self.model is None:
            raise ValueError("Сначала создайте модель")
        return self.model.summary()
    
    def save_model(self, filepath='models/final_model.h5'):
        """Сохранение модели"""
        if self.model is None:
            raise ValueError("Модель не создана")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"✅ Модель сохранена: {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Загрузка сохранённой модели"""
        return keras.models.load_model(filepath)
    
    def get_layer_names(self):
        """Получить имена всех слоёв"""
        if self.model is None:
            raise ValueError("Модель не создана")
        return [layer.name for layer in self.model.layers]


if __name__ == "__main__":
    # Пример использования
    num_classes = 38  # PlantVillage имеет 38 классов
    
    model_builder = PlantDiseaseModel(
        num_classes=num_classes,
        img_size=(224, 224),
        model_type='efficientnetb3'
    )
    
    model = model_builder.build_model()
    model_builder.compile_model()
    
    print("\n📊 Архитектура модели:")
    model_builder.summary()
    
    print(f"\n✅ Параметров в модели: {model.count_params():,}")
