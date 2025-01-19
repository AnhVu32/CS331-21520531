from PIL import Image
import torch
from torchvision import transforms

embedding_size = 1280

def get_vector(model_img_layer_4, t_img, device):
    my_emb = torch.zeros(1, embedding_size, 7, 7)
    t_img = torch.autograd.Variable(t_img).to(device)
    
    def hook(model, input, output):
        my_emb.copy_(output.data)
        
    h = model_img_layer_4.register_forward_hook(hook)
    model_img_layer_4(t_img)
    h.remove()
    return my_emb

def extract_image_feature(model_img_layer_4, data, device):
    img = Image.open(data).convert('RGB')
    scaler = transforms.Resize([224, 224])
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    transform = transforms.ToTensor()
    
    img = normalizer(transform(scaler(img)))
    img = img.unsqueeze(0).to(device)
    img_emb = get_vector(model_img_layer_4, img, device)
    return img_emb
