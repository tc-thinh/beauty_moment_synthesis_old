import torch
import torchvision.transforms as T
from PIL import Image

from model import model

device = 'cpu'
model_path = 'model/SDD_FIQA_checkpoints_r50.pth'


def read_img(img_path):  # read image & data pre-process

    data = torch.randn(1, 3, 112, 112)

    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = Image.open(img_path).convert("RGB")
    data[0, :, :, :] = transform(img)

    return data


def network(model_path, device):
    net = model.R50([112, 112], use_type="Qua").to(device)
    net_dict = net.state_dict()
    data_dict = {
        key.replace('module.', ''): value for key, value in torch.load(model_path, map_location=device).items()}
    net_dict.update(data_dict)
    net.load_state_dict(net_dict)
    net.eval()

    return net


def FIQA(img_path):
    net = network(model_path, device)
    input_data = read_img(img_path)
    pred_score = net(input_data).data.cpu().numpy().squeeze()
    return pred_score
