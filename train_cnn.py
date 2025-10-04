import torch
import torch.nn as nn
import torch.optim as optim

# Import the functions from helper library
from helper_lib.model import get_model
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model

def main():
    """
    Main function to orchestrate the model training and evaluation process.
    """
    # 1. Configuration 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    EPOCHS = 10 
    MODEL_NAME = 'FinalCNN'  
    MODEL_SAVE_PATH = 'final_cnn_model.pth' 

    print(f"--- Configuration ---")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print("-" * 20)
    
    # 2. Load Data (CIFAR-10 will now be resized to 64x64)
    print("Loading data...")
    train_loader = get_data_loader(dataset='cifar10', batch_size=BATCH_SIZE, train=True)
    test_loader = get_data_loader(dataset='cifar10', batch_size=BATCH_SIZE, train=False)

    # 3. Initialize Model, Loss, and Optimizer 
    print(f"Initializing {MODEL_NAME}...")
    model = get_model(MODEL_NAME, num_classes=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Train the Model 
    trained_model = train_model(model, train_loader, criterion, optimizer, DEVICE, EPOCHS)

    # 5. Evaluate the Model 
    print("\n--- Evaluating Model ---")
    evaluate_model(trained_model, test_loader, DEVICE)

    # 6. Save the Trained Model 
    print(f"\n--- Saving Model ---")
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()