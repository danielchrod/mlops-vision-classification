import yaml
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from .utils import FoodDataset, get_transforms 

def train_model():

    with open("config.yaml", "r") as f:
        config_data = yaml.safe_load(f)

    wandb.init(
        project=config_data["project_name"],
        config=config_data["hyperparameters"]
    )

    train_transforms = get_transforms(img_size=256, is_train=True)
    
    train_dataset = FoodDataset(data_dir="data/train", transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.DeepLabV3Plus(encoder_name="resnet50", classes=104).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(wandb.config.epochs):
        model.train()
        epoch_loss = 0
        
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{wandb.config.epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch}) 
        print(f"Epoch {epoch+1} finalizada. Loss: {avg_loss}")

    torch.save(model.state_dict(), "models/mejor_modelo_B.pth")
    wandb.finish()

if __name__ == "__main__":
    train_model()