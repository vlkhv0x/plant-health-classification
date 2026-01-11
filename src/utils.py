"""
Вспомогательные функции для проекта классификации болезней растений
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


def create_directories():
    """Создание необходимых директорий"""
    dirs = ['data/raw', 'data/processed', 'models', 'reports', 'logs', 'notebooks']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Директории созданы")


def plot_training_history(history, save_path='reports/training_history.png'):
    """Визуализация истории обучения"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[0, 1].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-3 Accuracy
    if 'top_3_accuracy' in history.history:
        axes[1, 0].plot(history.history['top_3_accuracy'], label='Train', linewidth=2)
        axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Val', linewidth=2)
        axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Precision/Recall
    if 'precision' in history.history:
        axes[1, 1].plot(history.history['precision'], label='Train Precision', linewidth=2)
        axes[1, 1].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
            axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Precision & Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ График обучения сохранён: {save_path}")
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path='reports/confusion_matrix.png'):
    """Визуализация матрицы ошибок"""
    if isinstance(cm, list):
        cm = np.array(cm)
    
    # Для большого количества классов уменьшим размер
    n_classes = len(class_names)
    figsize = (max(12, n_classes * 0.4), max(10, n_classes * 0.35))
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Нормализация
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Percentage'})
    
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Поворачиваем подписи для читаемости
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrix сохранена: {save_path}")
    plt.close()


def plot_sample_predictions(model, test_generator, class_names, 
                           num_samples=16, save_path='reports/predictions_sample.png'):
    """Визуализация примеров предсказаний"""
    test_generator.reset()
    images, true_labels = next(test_generator)
    
    images = images[:num_samples]
    true_labels = true_labels[:num_samples]
    
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels_idx = np.argmax(true_labels, axis=1)
    
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i])
        ax.axis('off')
        
        true_class = class_names[true_labels_idx[i]]
        pred_class = class_names[predicted_labels[i]]
        confidence = predictions[i][predicted_labels[i]]
        
        # Сокращаем длинные названия
        true_class_short = true_class.replace('___', '\n')[:30]
        pred_class_short = pred_class.replace('___', '\n')[:30]
        
        color = 'green' if true_labels_idx[i] == predicted_labels[i] else 'red'
        title = f"True: {true_class_short}\nPred: {pred_class_short}\n({confidence:.2%})"
        ax.set_title(title, fontsize=8, color=color, fontweight='bold')
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Примеры предсказаний сохранены: {save_path}")
    plt.close()


def save_training_config(model_type, num_classes, img_size, batch_size, 
                        epochs, learning_rate, use_class_weights, class_names):
    """Сохранение конфигурации обучения"""
    config = {
        'model_type': model_type,
        'num_classes': num_classes,
        'img_size': img_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'use_class_weights': use_class_weights,
        'class_names': class_names
    }
    
    with open('models/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("✅ Конфигурация сохранена: models/config.json")
    return config
