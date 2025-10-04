import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.model import get_model
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model

def main():
    """
    Main function to orchestrate the model training and evaluation process.
    """
    # Configuration 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    EPOCHS = 10  
    MODEL_NAME = 'EnhancedCNN'  
    MODEL_SAVE_PATH = 'enhanced_cnn_model.pth'

    print(f"--- Configuration ---")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print("-" * 20)
    
    # Load Data 
    print("Loading data...")
    train_loader = get_data_loader(batch_size=BATCH_SIZE, train=True)
    test_loader = get_data_loader(batch_size=BATCH_SIZE, train=False)

    # Initialize Model, Loss, and Optimizer 
    print(f"Initializing {MODEL_NAME}...")
    model = get_model(MODEL_NAME, num_classes=10) # CIFAR-10 has 10 classes

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the Model 
    trained_model = train_model(model, train_loader, criterion, optimizer, DEVICE, EPOCHS)

    # Evaluate the Model 
    print("\n--- Evaluating Model ---")
    evaluate_model(trained_model, test_loader, DEVICE)

    # Save the Trained Model 
    print(f"\n--- Saving Model ---")
    # Save the model's learned weights to a file
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()