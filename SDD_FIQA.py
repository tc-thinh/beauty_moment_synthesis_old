import os
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from misc.extract_bbox import *
from model import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = 'model/SDD_FIQA_checkpoints_r50.pth'


def process_fiqa_image(img):  # read image & data pre-process

    data = torch.randn(1, 3, 112, 112)

    transform = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    data[0, :, :, :] = transform(im_pil)

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


def FIQA(df, path):
    net = network(model_path, device)
    fiqa_score = []
    bbox = []
    filename = []
    for i in range(len(df)):
        if df["bboxes"][i][0] is not None:
            input_data = get_target_bbox(os.path.join(path, df["filename"][i]), df["bboxes"][i], p=0.15)
            score = []
            for j in input_data:
                if j.shape[0] > 0 and j.shape[1] > 0:
                    img = process_fiqa_image(j).to(device)
                    pred_score = net(img).data.cpu().numpy().squeeze()
                    score.append(pred_score)
        try:
            if max(score) > 40:
                filename.append(df["filename"][i])
                bbox.append(df["bboxes"][i])
                fiqa_score.append(score)
        except:
            pass

    new_df = pd.DataFrame({'filename': filename, 'bboxes': bbox, "fiqa_score": fiqa_score})
    return new_df
