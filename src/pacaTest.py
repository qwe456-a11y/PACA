import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
import neural_renderer as nr

from utils.data_loader import MyDatasetTestAdv
from utils.network import FIR_Net, Generator


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    texture_size = 6
    input_size = 800

    # Load OBJ
    vertices, faces, textures = nr.load_obj(
        filename_obj=args.obj,
        texture_size=texture_size,
        normalization=False,
        load_texture=True
    )

    vertices, faces, textures = vertices.to(device), faces.to(device), textures.to(device)

    # Adversarial Texture
    texture_content_adv = torch.from_numpy(np.load(args.textures)).float().to(device)
    texture_origin = textures.unsqueeze(0)

    texture_mask = np.zeros(
        (faces.shape[0], texture_size, texture_size, texture_size, 3),
        dtype=np.float32
    )

    with open(args.faces, 'r') as f:
        for face_id in f:
            if face_id.strip():
                texture_mask[int(face_id.strip()) - 1] = 1

    texture_mask = torch.from_numpy(texture_mask).float().to(device).unsqueeze(0)

    def cal_texture(texture_content):
        textures_new = 0.5 * (torch.tanh(texture_content) + 1)
        return texture_origin * texture_mask + (1 - texture_mask) * textures_new

    textures_adv = cal_texture(texture_content_adv)

    # Dataset
    data_dir = os.path.join(args.datapath, 'train/')
    mask_dir = os.path.join(args.datapath, 'masks/')
    label_dir = os.path.join(args.datapath, 'train_label_new/')

    dataset = MyDatasetTestAdv(data_dir, input_size, texture_size, faces, vertices, distence=None, mask_dir=mask_dir, ret_mask=True, label_dir=label_dir)

    dataset.set_textures(textures_adv)

    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False)

    # Generator
    attu_net = FIR_Net().to(device)
    generator = Generator(feature_extractor=attu_net).to(device)
    generator.load_state_dict(torch.load(args.generator, map_location=device))
    generator.eval()

    # Detector (optional)
    detector = None
    if args.detector:
        detector = torch.hub.load('./src', 'custom', args.detector, source='local', force_reload=True).to(device)
        detector.eval()

    total_images = 0
    attack_success = 0

    print("Start Testing...")

    for i, (index, total_img, texture_img, mask, filename, img, label) in enumerate(tqdm(loader)):

        batch_size = img.size(0)
        total_images += batch_size

        img = img.to(device)
        texture_img = texture_img.to(device)
        mask = mask.to(device)


        label = label.squeeze(0)  
        if label.ndim == 1:
            label = label.unsqueeze(0)  

        label_class = label[:, 0] 
        label_boxes = label[:, 1:] * input_size 

        x_center, y_center, width, height = label_boxes[:, 0], label_boxes[:, 1], label_boxes[:, 2], label_boxes[:, 3]
        label_boxes = torch.stack((
            x_center - width / 2,
            y_center - height / 2,
            x_center + width / 2,
            y_center + height / 2
        ), dim=1)

        img_cut = img * mask

        generated_img, _, _ = generator(img_cut, texture_img, mask)

        generated_imgs=(1 - mask) * img + (255 * generated_img) * mask
        img_np =  generated_imgs.data.cpu().numpy()[0]
        img_np = Image.fromarray(np.transpose(img_np, (1, 2, 0)).astype('uint8'))
        filename_base = filename[0].split('.')[0]
        # img_np.save(fr'{args.output_dir}/{filename_base}.png')

        if detector:

            results = detector(img_np)
            boxes = results.xyxy[0][:, :4] 
            scores = results.xyxy[0][:, 4]
            classes = results.xyxy[0][:, 5]
            valid_indices = scores > 0.5
            valid_boxes = boxes[valid_indices]
            valid_predicted_classes = classes[valid_indices]
            valid_scores = scores[valid_indices]

            detected_flags = []

            for j in range(label_boxes.shape[0]):
                true_box = label_boxes[j].cpu().numpy()
                true_class = int(label_class[j].cpu().numpy())
                found = False
                for k in range(valid_boxes.shape[0]):
                    pred_box = valid_boxes[k].cpu().numpy()
                    pred_class = int(valid_predicted_classes[k].cpu().numpy())
                    if pred_class ==true_class and calculate_iou(pred_box, true_box) > 0.5:
                        found = True
                        break
                detected_flags.append(found)

            if not any(detected_flags):
                attack_success += 1
            
            
            img_with_boxes = results.render()[0]
            img_with_boxes_pil = Image.fromarray(img_with_boxes)
            img_with_boxes_pil.save(fr'{args.output_dir}/{filename_base}.png')

    print("=================================")
    print("Total Images:", total_images)

    if detector and total_images > 0:
        print(f"Attack Success Rate (ASR): {attack_success / total_images * 100:.2f}%")

    print("=================================")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("PACA Test Script")

    parser.add_argument("--generator", required=True)
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--faces", required=True)
    parser.add_argument("--textures", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--detector", default=None)

    args = parser.parse_args()
    main(args)