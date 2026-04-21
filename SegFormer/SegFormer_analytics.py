import json
import matplotlib.pyplot as plt
with open('SegFormer/results/best_model/SegFormer_analytics.json', 'r') as f:
    data = json.load(f)
epochs = [item['epoch'] for item in data]
train_loss = [item['train']['acc'] for item in data]
test_loss = [item['test']['acc'] for item in data]
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Accuracy', marker='o', linestyle='-', color='blue')
plt.plot(epochs, test_loss, label='Test Accuracy', marker='o', linestyle='-', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs. Test Accuracy per Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('acc_comparison.png')
plt.show()