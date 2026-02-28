import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import cv2
import numpy as np
import cupy as cp
import warnings
import chainer
from PIL import Image
import math
warnings.filterwarnings("ignore")
sys.path.append("./neural_renderer/")
import matplotlib.pyplot as plt
import utils.nmr_test as nmr
import neural_renderer
from torchvision import transforms
from torchvision.transforms import functional as F

class MyDatasetTestAdv(Dataset):
    def __init__(self, data_dir, img_size, texture_size, faces, vertices, distence=None, mask_dir='', ret_mask=False, label_dir=''):
        self.data_dir = data_dir
        self.files = []
        files = os.listdir(data_dir)
        for file in files:
            if distence is None:
                self.files.append(file)
            # 计算车辆和相机之间的距离，并将距离小于等于 distence 的文件名添加到 self.files 列表中
            else:
                data = np.load(os.path.join(self.data_dir, file))
                obj_location = data["obj_location"]
                cam_location = data["cam_location"]
                dis = cam_location - obj_location
                dis = np.sum(dis ** 2)
                # print(dis)
                if dis <= distence:
                    self.files.append(file)
        print(len(self.files))
        self.augment = True
        self.img_size = img_size
        textures = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
        self.textures_adv = torch.from_numpy(textures).cuda(device=0)
        self.faces_var = faces[None, :, :]
        self.vertices_var = vertices[None, :, :]
        self.mask_renderer = nmr.NeuralRenderer(img_size=img_size).cuda()
        self.mask_dir = mask_dir
        self.ret_mask = ret_mask
        self.label_dir = label_dir
        self.mask_renderer.renderer.renderer.camera_mode = "projection"
        self.mask_renderer.renderer.renderer.K = None
        self.mask_renderer.renderer.renderer.R = None
        self.mask_renderer.renderer.renderer.t = None
        self.mask_renderer.renderer.renderer.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]])
        self.mask_renderer.renderer.renderer.orig_size = 800
        self.mask_renderer.renderer.renderer.light_direction = [0, 0, -1]
        self.mask_renderer.renderer.renderer.camera_up = [0, 0, 1]
        self.mask_renderer.renderer.renderer.background_color = [1, 1, 1]

    def set_textures(self, textures_adv):
        self.textures_adv = textures_adv

    def __getitem__(self, index):
        file = os.path.join(self.data_dir, self.files[index])
        data = np.load(file, allow_pickle=True)  #.item()
        img = data['img']
        # 获取车辆和相机的变换矩阵
        # obj_trans, cam_trans = data['object'], data['cam_trans']
        obj_location = data["obj_location"]
        obj_rotation = data["obj_rotation"]
        cam_location = data["cam_location"]
        cam_rotation = data["cam_rotation"]

        # K = np.array([[888.88889, 0, 320],
        #       [0, 888.88889, 320],
        #       [0, 0, 1]])
        K = np.array([[1111.11111, 0, 400],
            [0, 1111.11111, 400],
            [0, 0, 1]]) 
        K = np.tile(K.reshape(1, 3, 3), (1, 1, 1))
        R, t = camera_pose_in_world(obj_location, obj_rotation, cam_location, cam_rotation)
        # print("R:", R)
        # print("t:", t)

        # 将 NumPy 数组转换为 PyTorch 张量，并发送到指定设备上
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        K = torch.tensor(K, dtype=torch.float32, device=device)
        R = torch.tensor(R, dtype=torch.float32, device=device)
        t = torch.tensor(t, dtype=torch.float32, device=device)
        # 更新渲染器的相机参数，以便渲染器能够生成正确的图像
        self.mask_renderer.renderer.renderer.K = K
        self.mask_renderer.renderer.renderer.R = R
        self.mask_renderer.renderer.renderer.t = t

        imgs_pred = self.mask_renderer.forward(self.vertices_var, self.faces_var, self.textures_adv)

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = torch.from_numpy(img).to(device)

        imgs_pred = imgs_pred / torch.max(imgs_pred)
        label_file = os.path.join(self.label_dir, "%s.txt" % self.files[index][:-4])
        with open(label_file, 'r') as file:
            lines = file.readlines()  # 读取所有行
        if lines:
            # 将第一行数据按空格分割，并转换为浮点数
            first_line = np.array([float(x) for x in lines[0].split()])
            # 转换为 PyTorch 张量
            labels_out = torch.from_numpy(first_line)
        else:
            # 如果文件为空，初始化空张量
            labels_out = torch.empty(0)

        if self.ret_mask:
            mask_file = os.path.join(self.mask_dir, "%s.png" % self.files[index][:-4])
            mask = Image.open(mask_file).convert('L')
            mask = mask.resize((self.img_size, self.img_size), Image.ANTIALIAS)
            mask = np.array(mask)
            # 使用 Otsu's 方法进行二值化处理，确保边缘平滑
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 确保掩码是二值的
            mask = mask > 127
            mask = torch.from_numpy(mask.astype('float32')).to(device)

            total_img = (1 - mask) * img + (255 * imgs_pred) * mask
            imgs_pred= mask * imgs_pred
            
            # img1 = img.clone()
            # total_img1 = total_img.clone()
            imgs_pred1 = imgs_pred.clone()
            # mask1 = mask.clone()
            # Image.fromarray(np.transpose(img.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8')).save(
            #         os.path.join('./src/test/json', 'img.png'))
            # Image.fromarray(np.transpose(total_img.data.cpu().numpy()[0], (1, 2, 0)).astype('uint8')).save(
            #         os.path.join('./src/test/json', 'test_total.png'))
            # Image.fromarray(
            #     imgs_pred1.data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
            #     os.path.join('./src/test/json', 'texture2.png'))
            # Image.fromarray((255 * mask1).data.cpu().numpy().astype('uint8')).save(
            #     os.path.join('./src/test/json', 'mask.png'))
            return index, total_img.squeeze(0), imgs_pred.squeeze(0), mask, self.files[index], img.squeeze(0), labels_out
        total_img = img + 255 * imgs_pred
        return index, total_img.squeeze(0), imgs_pred.squeeze(0), self.files[index], img.squeeze(0), labels_out

    def __len__(self):
        return len(self.files)
    
def quaternion_to_rotation_matrix(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    
    # Compute rotation matrix
    R = np.array([[1 - 2 * (y**2 + z**2), 2 * (x*y - w*z), 2 * (x*z + w*y)],
                [2 * (x*y + w*z), 1 - 2 * (x**2 + z**2), 2 * (y*z - w*x)],
                [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x**2 + y**2)]])
    return R

def camera_pose_in_world(object_pos, object_quat, camera_pos, camera_quat):
    # Object to world transform
    R_obj_to_world = quaternion_to_rotation_matrix(object_quat)
    t_obj_to_world = np.array(object_pos)
    
    # Camera to world transform
    R_cam_to_world = quaternion_to_rotation_matrix(camera_quat)
    t_cam_to_world = np.array(camera_pos)
    
    # World to camera transform (inverse of camera to world)
    R_world_to_cam = R_cam_to_world.T
    t_world_to_cam = -R_world_to_cam @ t_cam_to_world
    
    # Object to camera transform
    R = R_world_to_cam @ R_obj_to_world
    t = R_world_to_cam @ t_obj_to_world + t_world_to_cam

    # 将旋转矩阵 R 调整为目标格式 (batch_size, 3, 3)
    R = np.tile(R.reshape(1, 3, 3), (1, 1, 1))
    # 将平移向量 t 调整为目标格式 (batch_size, 1, 3)
    t = np.expand_dims(t, axis=0)
    
    return R, t

def letterbox(img, new_shape=(800, 800), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

if __name__ == '__main__':
    obj_file = 'audi_et_te.obj'
    vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, load_texture=True)
    rnder = neural_renderer.Renderer()
    vertices = np.expand_dims(vertices, axis=0)
    faces = np.expand_dims(faces, axis=0)
    textures = np.expand_dims(textures, axis=0)
    faces = chainer.Variable(chainer.cuda.to_gpu(faces, 0))
    vertices = chainer.Variable(chainer.cuda.to_gpu(vertices, 0))
    textures = chainer.Variable(chainer.cuda.to_gpu(textures, 0))
    image = rnder.render(vertices, faces, textures)
    image = image.data[0]
    image = (np.clip(cp.asnumpy(image),0,1) * 255).astype(np.uint8)
    image = Image.fromarray(np.transpose(image, (1,2,0)))
    image.show()
    dataset = MyDatasetTestAdv('./src/carla_dataset/phy_attack/train/', 608, 4, faces, vertices)
    loader = DataLoader(
        dataset=dataset,   
        batch_size=3,     
        shuffle=True,            
        #num_workers=2,              
    )
    
    for img, car_box in loader:
        print(img.size(), car_box.size())
