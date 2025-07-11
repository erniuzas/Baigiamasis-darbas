import matplotlib.pyplot as plt


def plot_training_results(history1, history2=None, labels=("Model 1", "Model 2"), title="Modelių palyginimas"):
    plt.figure(figsize=(14, 5))

    # Tikslumo palyginimas
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['accuracy'], label=f'{labels[0]} - Train', color='blue')
    plt.plot(history1.history['val_accuracy'], label=f'{labels[0]} - Val', color='skyblue')
    
    if history2:
        plt.plot(history2.history['accuracy'], label=f'{labels[1]} - Train', color='green')
        plt.plot(history2.history['val_accuracy'], label=f'{labels[1]} - Val', color='lime')
    
    plt.title('Tikslumo (Accuracy) palyginimas')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Praradimo palyginimas
    plt.subplot(1, 2, 2)
    plt.plot(history1.history['loss'], label=f'{labels[0]} - Train', color='red')
    plt.plot(history1.history['val_loss'], label=f'{labels[0]} - Val', color='salmon')
    
    if history2:
        plt.plot(history2.history['loss'], label=f'{labels[1]} - Train', color='purple')
        plt.plot(history2.history['val_loss'], label=f'{labels[1]} - Val', color='violet')
    
    plt.title('Praradimo (Loss) palyginimas')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()