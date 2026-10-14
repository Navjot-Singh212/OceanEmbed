import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim


from model import OceanForecaster
from dataset import OceanDataset

def train_model():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    print("Loading dataset~")
    full_dataset=OceanDataset(normalised_dir="data/normalized",sequence_length=7)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader=DataLoader(train_dataset,batch_size=4,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False) 

    model=OceanForecaster(input_channels=7,hidden_channels=32,output_channels=15).to(device)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()    #does the actual updation of weights
            
            total_loss+=loss.item()

            if ((batch_idx % 10)==0):
                print(f"Epoch [{epoch+1}/{epochs}] | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        model.eval()
        val_loss=0.0
        with torch.no_grad():
            for inputs,targets in val_loader:
                inputs,targets=inputs.to(device), targets.to(device)
                prediction=model(inputs)
                loss=criterion(prediction,targets)
                val_loss+=loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    torch.save(model.state_dict(), "data/ocean_model.pth")
    print("Training complete! Model saved to ocean_model.pth")

if __name__ == "__main__":
    train_model()