"""
Скрипт для предсказания болезней растений на новых изображениях
"""

import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append('src')

from data_preprocessing import load_and_preprocess_image


def parse_args():
    parser = argparse.ArgumentParser(description='Предсказание болезней растений')
    parser.add_argument('--image_path', type=str, required=True, help='Путь к изображению')
    parser.add_argument('--model_path', type=str, default='models/best_model.h5')
    parser.add_argument('--config_path', type=str, default='models/config.json')
    parser.add_argument('--top_k', type=int, default=5, help='Топ-K предсказаний')
    parser.add_argument('--show_image', action='store_true', help='Показать изображение')
    return parser.parse_args()


def predict_disease(model, image_path, class_names, img_size=(224, 224), top_k=5):
    """Предсказание для изображения"""
    img = load_and_preprocess_image(image_path, img_size)
    img_batch = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img_batch, verbose=0)[0]
    
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_probs = [predictions[i] for i in top_indices]
    
    return {
        'top_classes': top_classes,
        'top_probabilities': [float(p) for p in top_probs],
        'all_predictions': {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    }, img


def visualize_prediction(image, predictions, save_path=None):
    """Визуализация предсказания"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Изображение листа', fontsize=14, fontweight='bold')
    
    top_classes = predictions['top_classes']
    top_probs = predictions['top_probabilities']
    
    # Сокращаем названия
    short_names = [c.replace('___', '\n')[:40] for c in top_classes]
    
    y_pos = np.arange(len(top_classes))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_classes)))
    
    bars = ax2.barh(y_pos, top_probs, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(short_names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Вероятность', fontsize=12)
    ax2.set_title('Топ предсказаний', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax2.text(prob + 0.02, i, f'{prob:.2%}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Визуализация сохранена: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("🔮 ПРЕДСКАЗАНИЕ БОЛЕЗНЕЙ РАСТЕНИЙ")
    print("=" * 80)
    
    if not os.path.exists(args.image_path):
        print(f"❌ Изображение не найдено: {args.image_path}")
        return
    
    print(f"\n📷 Изображение: {args.image_path}")
    
    # Загрузка конфигурации
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    class_names = config['class_names']
    img_size = config.get('img_size', 224)
    
    # Загрузка модели
    print(f"\n📥 Загрузка модели: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    print("✅ Модель загружена")
    
    # Предсказание
    print(f"\n🔮 Выполнение предсказания (top-{args.top_k})...")
    predictions, image = predict_disease(
        model, args.image_path, class_names,
        img_size=(img_size, img_size), top_k=args.top_k
    )
    
    # Вывод результатов
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ")
    print("=" * 80)
    
    print(f"\n🥇 Топ-{args.top_k} предсказаний:")
    for i, (cls, prob) in enumerate(zip(predictions['top_classes'], 
                                        predictions['top_probabilities']), 1):
        plant_disease = cls.replace('___', ' → ')
        print(f"   {i}. {plant_disease:<50} {prob:.2%}")
    
    print(f"\n🎯 Наиболее вероятный диагноз:")
    best_class = predictions['top_classes'][0]
    best_prob = predictions['top_probabilities'][0]
    plant, disease = best_class.split('___')
    print(f"   Растение: {plant.replace('_', ' ')}")
    print(f"   Состояние: {disease.replace('_', ' ')}")
    print(f"   Уверенность: {best_prob:.2%}")
    
    # Визуализация
    if args.show_image:
        save_path = f"reports/prediction_{os.path.basename(args.image_path)}.png"
        visualize_prediction(image, predictions, save_path=save_path)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
