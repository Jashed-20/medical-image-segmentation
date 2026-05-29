#train test split

import albumentations as A
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
])


train_dataset = MRIDataset(train_image_paths, train_mask_paths, img_size=256, transform=train_transform)
test_dataset = MRIDataset(test_image_paths, test_mask_paths, img_size=256, transform=None)


train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

print(f"Total training samples: {len(train_dataset)}")
print(f"Total testing samples: {len(test_dataset)}")



from torch.utils.data import DataLoader

dataset = MRIDataset(image_paths, mask_paths)

train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet50_UNet(num_classes=1).to(device)


#calculating loss



class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        smooth = 1e-6
        intersection = (preds * targets).sum()
        dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

        return 1 - dice


bce = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()

def combined_loss(preds, targets):
    return bce(preds, targets) + dice_loss(preds, targets)


import torch.optim as optim

# Map your custom function to the 'criterion' variable used in the loop
criterion = combined_loss

# Setup the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#training the model
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    model.train()

    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)


        outputs = model(images)


        loss = criterion(outputs, masks)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} completed | Avg Loss: {total_loss/len(train_loader):.4f}")




