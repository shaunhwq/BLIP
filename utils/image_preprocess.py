from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def load_image(image, image_size, device):
    w, h = image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image