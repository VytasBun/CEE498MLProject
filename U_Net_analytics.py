import json
import matplotlib.pyplot as plt

# Load the data from your JSON file
with open('U-Net_Pavement_Cracking/results/Early_Stop/unet_analytics_earlystop.json', 'r') as f:
    data = json.load(f)

# Extract epoch numbers and loss values
epochs = [item['epoch'] for item in data]
train_loss = [item['train']['loss'] for item in data]
test_loss = [item['test']['loss'] for item in data]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', marker='o', linestyle='-', color='blue')
plt.plot(epochs, test_loss, label='Test Loss', marker='o', linestyle='-', color='orange')

# Add labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Test Loss per Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Show or save the plot
plt.savefig('loss_comparison_earlystop.png')
plt.show()