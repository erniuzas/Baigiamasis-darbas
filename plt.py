# ========== 1. Spalvoto modelio grafikai ==========
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history_color.history['accuracy'], label='Train Accuracy', color='green')
plt.plot(history_color.history['val_accuracy'], label='Validation Accuracy', color='blue')
plt.title('Spalvoto modelio tikslumas')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_color.history['loss'], label='Train Loss', color='red')
plt.plot(history_color.history['val_loss'], label='Validation Loss', color='black')
plt.title('Spalvoto modelio praradimas')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.suptitle("Spalvotų paveikslėlių modelio rezultatai", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ========== 2. Nespalvoto (grayscale) modelio grafikai ==========
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history_gray.history['accuracy'], label='Train Accuracy', color='orange')
plt.plot(history_gray.history['val_accuracy'], label='Validation Accuracy', color='purple')
plt.title('Grayscale modelio tikslumas')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_gray.history['loss'], label='Train Loss', color='purple')
plt.plot(history_gray.history['val_loss'], label='Validation Loss', color='brown')
plt.title('Grayscale modelio praradimas')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.suptitle("Nespalvotų paveikslėlių modelio rezultatai", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print(f"Viso paveikslėlių: {len(X_color)}")
print(f"Mokymui: {len(X_color_train)}, Testavimui: {len(X_color_test)}")