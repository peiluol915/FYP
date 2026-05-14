import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time

try:
    from models import OAN, ArcMarginProduct
except ModuleNotFoundError:
    from backend.models import OAN, ArcMarginProduct

class OANClassifier(nn.Module):
    def __init__(self, num_classes, backbone_type='resnet50', use_cbam=True):
        super(OANClassifier, self).__init__()
        self.oan = OAN(backbone_type=backbone_type, use_cbam=use_cbam)
        self.arcface = ArcMarginProduct(512, num_classes)
        
    def forward(self, x, labels=None):
        embeddings = self.oan(x)
        if labels is not None:
            return self.arcface(embeddings, labels)
        return embeddings

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting OAN ({args.backbone}) training on: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    full_dataset = datasets.ImageFolder(root=args.data, transform=transform)
    num_classes = len(full_dataset.classes)
    print(f"Detected {num_classes} identities/classes in dataset.")

    # Train / Val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    # Update transform for val dataset manually (random_split uses the same dataset reference)
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    model = OANClassifier(num_classes, backbone_type=args.backbone, use_cbam=args.use_cbam).to(device)
    
    if args.resume and os.path.exists(args.resume):
        model.oan.load_state_dict(torch.load(args.resume, map_location=device), strict=False)
        print(f"Resumed from {args.resume}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    os.makedirs("weights", exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs, labels)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 20 == 0:
                print(f"[Epoch {epoch + 1}/{args.epochs}, Batch {i + 1}/{len(train_loader)}] Loss: {running_loss / 20:.4f} Acc: {100.*correct/total:.2f}%")
                running_loss = 0.0
                
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = model(inputs, labels)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = 100. * val_correct / max(1, val_total)
        epoch_time = time.time() - start_time
        
        print(f"End of Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.1f}s")
        
        # Check if running under newer PyTorch version
        try:
            scheduler.step(avg_val_loss)
        except TypeError:
            scheduler.step()
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"weights/oan_{args.backbone}_best.pth"
            torch.save(model.oan.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
            
    final_save_path = f"weights/oan_{args.backbone}_final.pth"
    torch.save(model.oan.state_dict(), final_save_path)
    print(f"Training complete. Final weights saved to: {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the OAN Backbone")
    parser.add_argument("--data", type=str, required=True, help="Path to masked training dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "mobilefacenet"], help="Backbone type")
    parser.add_argument("--no-cbam", action="store_false", dest="use_cbam", help="Disable CBAM attention module")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    train(args)
