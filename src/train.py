import yaml
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

def train_model():

    with open("config.yaml", "r") as f:
        config_data = yaml.safe_load(f)

    wandb.init(
        project=config_data["project_name"],
        config=config_data["hyperparameters"]
    )


    # SIMULACIÓN DE DATOS 
    # Creamos 4 imágenes de tamaño 3x256x256 y sus máscaras
    dummy_imgs = torch.randn(4, 3, 256, 256)
    dummy_masks = torch.randint(0, 104, (4, 256, 256)) 
    
    dataset = TensorDataset(dummy_imgs, dummy_masks)
    train_loader = DataLoader(dataset, batch_size=wandb.config.batch_size)

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
        print(f"Epoch {epoch} finalizada. Loss: {avg_loss}")

    torch.save(model.state_dict(), "models/mejor_modelo_B.pth")
    wandb.finish()

if __name__ == "__main__":
    train_model()