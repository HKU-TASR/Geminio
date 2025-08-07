from PIL import Image
import torch
import tqdm
from transformers import CLIPProcessor, CLIPModel
import torchvision

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
clip_name = "openai/clip-vit-large-patch14"
batch_size = 10

processor = CLIPProcessor.from_pretrained(clip_name)
model = CLIPModel.from_pretrained(clip_name).to(device)

dataset = torchvision.datasets.ImageNet(root='./data', split='val')
print('# of Samples: %d' % len(dataset.imgs))

buffer = []
for img_path, _ in tqdm.tqdm(dataset.imgs):
    inputs = processor(images=Image.open(img_path), return_tensors='pt')
    image_features = model.get_image_features(pixel_values=inputs['pixel_values'].to(device))
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    buffer.append(image_features.detach().cpu())
buffer = torch.cat(buffer, dim=0)
print(buffer.shape)

print('Saving to ./data/imagenet-clip-test.pt')
torch.save(buffer, './data/imagenet-clip-test.pt')

meta = {'class_embeds': []}
for cls in dataset.classes:
    tmp = processor(text=cls, return_tensors="pt")
    text_embeds = model.get_text_features(input_ids=tmp['input_ids'].to(device))
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    meta['class_embeds'].append(text_embeds.detach().cpu())
meta = {'class_embeds': torch.cat(meta['class_embeds'], dim=0)}
print({k: meta[k].shape for k in meta})
torch.save(meta, './data/imagenet-clip-meta.pt')
