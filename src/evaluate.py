"""
Скрипт для оценки модели классификации болезней растений
"""

import os
import sys
import argparse
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

sys.path.append('src')

from data_preprocessing import PlantDiseasePreprocessor
from utils import plot_confusion_matrix, plot_sample_predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Оценка модели')
    parser.add_argument('--model_path', type=str, default='models/best_model.h5')
    parser.add_argument('--data_dir', type=str, default='data/raw/PlantVillage')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--detailed_analysis', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("📊 ОЦЕНКА МОДЕЛИ")
    print("=" * 80)
    
    # Загрузка модели
    print(f"\n📥 Загрузка модели: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    print("✅ Модель загружена")
    
    # Загрузка конфигурации
    with open('models/config.json', 'r') as f:
        config = json.load(f)
    class_names = config['class_names']
    
    # Подготовка данных
    print("\n📊 Подготовка тестовых данных...")
    preprocessor = PlantDiseasePreprocessor(
        data_dir=args.data_dir,
        img_size=(config['img_size'], config['img_size']),
        batch_size=args.batch_size
    )
    
    image_paths, labels = preprocessor.load_data_paths()
    df = preprocessor.create_dataframe(image_paths, labels)
    train_df, val_df, test_df = preprocessor.split_data(df)
    _, _, test_gen = preprocessor.create_data_generators(train_df, val_df, test_df)
    
    # Оценка
    print("\n🔮 Получение предсказаний...")
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Метрики
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
    
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ")
    print("=" * 80)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    print(f"\nPrecision (weighted): {report['weighted avg']['precision']:.4f}")
    print(f"Recall (weighted): {report['weighted avg']['recall']:.4f}")
    print(f"F1-Score (weighted): {report['weighted avg']['f1-score']:.4f}")
    
    # Сохранение результатов
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    with open('reports/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Classification report в текстовый файл
    with open('reports/classification_report.txt', 'w') as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\n✅ Результаты сохранены")
    
    # Визуализации
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)
    plot_sample_predictions(model, test_gen, class_names)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
