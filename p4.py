import pandas as pd, numpy as np, torch, time
from pathlib import Path
from PIL import Image
from torchvision import models, transforms

ROOT = Path(r"D:\ML CEP\dataverse_files")
OUT = Path(r"D:\ML CEP\CEP_CODE")

# here i load the split file, i keep the same row order for the saved features
meta = pd.read_csv(OUT / "split.csv")

# here i make a dictionary from image_id to its file path (images are in 2 folders)
paths = {p.stem: p for p in ROOT.glob("HAM10000_images_part_*/*.jpg")}

# here i load resnet18 that was trained on imagenet
# i remove the last layer (fc) so that it gives 512 numbers per image
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model.eval()   # here i switch to evaluation mode, we are not training it

# here i set the image preprocessing: resize to 224x224, make tensor, normalize
# the mean and std below are fixed imagenet numbers, they are not learned from our data
prep = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# here i go through the images in batches and collect the features
batch_size = 64
ids = meta["image_id"].tolist()
features = []
start = time.time()

with torch.no_grad():   # here i turn off gradients because we only predict
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        imgs = [prep(Image.open(paths[x]).convert("RGB")) for x in batch_ids]
        batch = torch.stack(imgs)
        out = model(batch)
        features.append(out.numpy())
        if (i // batch_size) % 20 == 0:
            print(i, "of", len(ids), "done,", round(time.time() - start), "seconds")

features = np.vstack(features)
print("feature shape:", features.shape)
print("total time (seconds):", round(time.time() - start))

# here i save the features so i do not need to compute them again
np.save(OUT / "emb_resnet18.npy", features)