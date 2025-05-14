import os
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights

# --- AŞAMA 1: DATALOADER VE TRANSFORMS ---

def get_data_loaders(data_dir='dataset', batch_size=32, image_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_data.classes

# --- AŞAMA 2: RESNET50 MODELİNİ ÖZELLİK ÇIKARICI OLARAK HAZIRLAMA ---

def get_feature_extractor(device):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
    model.to(device)
    return model

# --- AŞAMA 3: ÖZELLİK ÇIKARIMI ---

def extract_features(data_loader, model, device):
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for images, lbls in data_loader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.view(outputs.size(0), -1)  # Flatten to 1D
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels

# --- AŞAMA 4: ANA BLOK (DOSYA KAYDI DAHİL) ---

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("📦 Veriler yükleniyor...")
    train_loader, val_loader, test_loader, class_names = get_data_loaders()

    print("🧠 ResNet50 modeli hazırlanıyor...")
    feature_extractor = get_feature_extractor(device)

    print("🔍 Train setten özellik çıkarılıyor...")
    train_X, train_y = extract_features(train_loader, feature_extractor, device)

    print("🔍 Validation setten özellik çıkarılıyor...")
    val_X, val_y = extract_features(val_loader, feature_extractor, device)

    print("🔍 Test setten özellik çıkarılıyor...")
    test_X, test_y = extract_features(test_loader, feature_extractor, device)

    print("💾 Özellikler kaydediliyor...")
    np.save("train_X.npy", train_X)
    np.save("train_y.npy", train_y)
    np.save("val_X.npy", val_X)
    np.save("val_y.npy", val_y)
    np.save("test_X.npy", test_X)
    np.save("test_y.npy", test_y)

    print("✅ Tüm işlemler tamamlandı. Özellikler başarıyla çıkarıldı.")
