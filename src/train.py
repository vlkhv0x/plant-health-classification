"""
Скрипт для обучения модели классификации болезней растений
"""

import os
import argparse
import json
import sys

sys.path.append('src')

from data_preprocessing import PlantDiseasePreprocessor
from model import PlantDiseaseModel
from utils import plot_training_history, create_directories, save_training_config


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Обучение модели классификации болезней растений'
    )
    
    parser.add_argument('--data_dir', type=str, 
                       default='data/raw/PlantVillage',
                       help='Путь к датасету PlantVillage')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Размер батча')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Размер изображения')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--model_type', type=str, default='efficientnetb3',
                       choices=['efficientnetb3', 'resnet50', 'densenet121', 'mobilenetv2'],
                       help='Тип базовой модели')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Использовать веса классов')
    parser.add_argument('--augment_rare_classes', action='store_true',
                       help='Усиленная аугментация для редких классов')
    parser.add_argument('--fine_tune', action='store_true',
                       help='Выполнить fine-tuning после обучения')
    parser.add_argument('--fine_tune_epochs', type=int, default=15,
                       help='Количество эпох для fine-tuning')
    parser.add_argument('--use_tensorboard', action='store_true',
                       help='Использовать TensorBoard')
    
    return parser.parse_args()


def main():
    """Основная функция обучения"""
    
    args = parse_args()
    
    print("=" * 80)
    print("🌿 ОБУЧЕНИЕ МОДЕЛИ КЛАССИФИКАЦИИ БОЛЕЗНЕЙ РАСТЕНИЙ")
    print("=" * 80)
    
    # Создание директорий
    create_directories()
    
    # Проверка данных
    if not os.path.exists(args.data_dir):
        print(f"\n❌ Датасет не найден: {args.data_dir}")
        print("\nВыполните одно из действий:")
        print("  1. Скачайте PlantVillage датасет:")
        print("     git clone https://github.com/spMohanty/PlantVillage-Dataset.git data/raw/")
        print("  2. Или запустите: python src/download_data.py")
        print("  3. Или запустите демо: python demo.py")
        return
    
    # ========== 1. ПОДГОТОВКА ДАННЫХ ==========
    print("\n" + "=" * 80)
    print("📊 ШАГ 1: ПОДГОТОВКА ДАННЫХ")
    print("=" * 80)
    
    preprocessor = PlantDiseasePreprocessor(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    train_gen, val_gen, test_gen, class_weights = preprocessor.prepare_data_pipeline(
        augment_rare_classes=args.augment_rare_classes
    )
    
    num_classes = len(train_gen.class_indices)
    
    # ========== 2. СОЗДАНИЕ МОДЕЛИ ==========
    print("\n" + "=" * 80)
    print("🏗️  ШАГ 2: СОЗДАНИЕ МОДЕЛИ")
    print("=" * 80)
    
    model_builder = PlantDiseaseModel(
        num_classes=num_classes,
        img_size=(args.img_size, args.img_size),
        model_type=args.model_type
    )
    
    model = model_builder.build_model(trainable_base=False)
    model_builder.compile_model(
        learning_rate=args.learning_rate,
        class_weights=class_weights if args.use_class_weights else None
    )
    
    print(f"\n✅ Модель создана: {args.model_type}")
    print(f"   Параметров: {model.count_params():,}")
    print(f"   Классов: {num_classes}")
    
    # ========== 3. ОБУЧЕНИЕ МОДЕЛИ ==========
    print("\n" + "=" * 80)
    print("🚀 ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 80)
    
    callbacks = model_builder.get_callbacks(
        checkpoint_path='models/best_model.h5',
        use_tensorboard=args.use_tensorboard
    )
    
    history = model_builder.train(
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        use_class_weights=args.use_class_weights
    )
    
    # Сохранение истории
    history_dict = {k: [float(v) for v in vals] 
                   for k, vals in history.history.items()}
    with open('reports/training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    print("\n✅ История обучения сохранена")
    
    # Визуализация
    plot_training_history(history, save_path='reports/training_history.png')
    
    # ========== 4. FINE-TUNING (опционально) ==========
    if args.fine_tune:
        print("\n" + "=" * 80)
        print("🔧 ШАГ 4: FINE-TUNING МОДЕЛИ")
        print("=" * 80)
        
        fine_tune_history = model_builder.fine_tune(
            train_generator=train_gen,
            val_generator=val_gen,
            epochs=args.fine_tune_epochs,
            unfreeze_layers=50,
            learning_rate=args.learning_rate / 10
        )
        
        # Сохранение истории fine-tuning
        ft_history_dict = {k: [float(v) for v in vals]
                          for k, vals in fine_tune_history.history.items()}
        with open('reports/fine_tune_history.json', 'w') as f:
            json.dump(ft_history_dict, f, indent=4)
        
        plot_training_history(
            fine_tune_history,
            save_path='reports/fine_tune_history.png'
        )
    
    # ========== 5. СОХРАНЕНИЕ МОДЕЛИ ==========
    print("\n" + "=" * 80)
    print("💾 ШАГ 5: СОХРАНЕНИЕ МОДЕЛИ")
    print("=" * 80)
    
    model_builder.save_model('models/final_model.h5')
    
    # Сохранение конфигурации
    config = save_training_config(
        model_type=args.model_type,
        num_classes=num_classes,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_class_weights=args.use_class_weights,
        class_names=list(train_gen.class_indices.keys())
    )
    
    # ========== ФИНАЛЬНЫЙ ОТЧЁТ ==========
    print("\n" + "=" * 80)
    print("📊 ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
    
    best_val_acc = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    best_val_top3 = max(history.history['val_top_3_accuracy'])
    
    print(f"\n📈 Лучшие результаты на валидации:")
    print(f"   Accuracy: {best_val_acc:.4f}")
    print(f"   Top-3 Accuracy: {best_val_top3:.4f}")
    print(f"   Loss: {best_val_loss:.4f}")
    
    print(f"\n📁 Сохранённые файлы:")
    print(f"   ✓ models/best_model.h5")
    print(f"   ✓ models/final_model.h5")
    print(f"   ✓ models/config.json")
    print(f"   ✓ data/processed/class_mapping.json")
    print(f"   ✓ reports/training_history.json")
    print(f"   ✓ reports/training_history.png")
    
    print("\n🎯 Следующие шаги:")
    print("   1. Оценка на тестовой выборке:")
    print("      python src/evaluate.py")
    print("   2. Предсказание для новых изображений:")
    print("      python src/predict.py --image_path path/to/leaf.jpg --show_image")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
