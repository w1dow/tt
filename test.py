import tensorflow as tf
import matplotlib.pyplot as plt
import json
with open('./history.json','r') as f:
    history=json.load(f)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.plot(history['loss'], label='Loss')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
plt.legend()
plt.savefig('graph.png')
plt.show()