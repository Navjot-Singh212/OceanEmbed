import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from model import OceanForecaster
from dataset import OceanDataset

def train_model():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    print("Loading dataset~")
    train_dataset=OceanDataset(normalised_dir="data/normalized",sequence_length=7)

    train_loader=DataLoader(train_dataset,batch_size=4,shuffle=True)

    model=OceanForecaster(input_channels=7,hidden_channels=16,output_channels=15).to(device)
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    epochs=10
    print("Starting Training~")
    for epoch in range(epochs):
        model.train()
        total_loss=0.0

        for batch_idx,(inputs,targets) in enumerate(train_loader):
            inputs=inputs.to(device)
            targets=targets.to(device)

            optimizer.zero_grad()

            predictions=model(inputs)
            loss=criterion(predictions,targets)
            loss.backward()     #calculates how to update the weights 
            optimizer.step()    #does the actual updation of weights
            
            total_loss+=loss.item()

            if ((batch_idx % 10)==0):
                print(f"Epoch [{epoch+1}/{epochs}] | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "data/ocean_model.pth")
    print("Training complete! Model saved to ocean_model.pth")

if __name__ == "__main__":
    train_model()