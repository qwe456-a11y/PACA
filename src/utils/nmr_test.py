from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ntpath
import numpy as np
import scipy.misc
import math
import torch
import neural_renderer
from glob import glob
import os
import imageio

#############
### Utils ###
#############

# Convert the source tensor to the same type as the target tensor
# and ensure they are on the same device
def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src


# Compute camera position, direction, and up vector
# based on given camera and vehicle transformation parameters
def get_params(carlaTcam, carlaTveh):
    scale = 0.70

    # Determine camera position
    eye = [0, 0, 0]
    for i in range(0, 3):
        eye[i] = carlaTcam[0][i] * scale

    # Pitch angle
    pitch = math.radians(carlaTcam[1][0])
    # Yaw angle
    yaw = math.radians(carlaTcam[1][1])
    # Roll angle
    roll = math.radians(carlaTcam[1][2])

    # Compute camera direction from pitch and yaw
    cam_direct = [
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch)
    ]

    # Compute camera up direction from pitch and yaw
    cam_up = [
        math.cos(math.pi / 2 + pitch) * math.cos(yaw),
        math.cos(math.pi / 2 + pitch) * math.sin(yaw),
        math.sin(math.pi / 2 + pitch)
    ]

    # Camera position
    p_cam = eye

    # Camera look-at position
    p_dir = [
        eye[0] + cam_direct[0],
        eye[1] + cam_direct[1],
        eye[2] + cam_direct[2]
    ]

    # Camera up direction position
    p_up = [
        eye[0] + cam_up[0],
        eye[1] + cam_up[1],
        eye[2] + cam_up[2]
    ]

    p_l = [p_cam, p_dir, p_up]

    # Store transformed positions
    trans_p = []
    for p in p_l:
        # Check whether horizontal distance is zero
        if math.sqrt(p[0] ** 2 + p[1] ** 2) == 0:
            cosfi = 0
            sinfi = 0
        else:
            # Compute cosine and sine
            cosfi = p[0] / math.sqrt(p[0] ** 2 + p[1] ** 2)
            sinfi = p[1] / math.sqrt(p[0] ** 2 + p[1] ** 2)

        cossum = cosfi * math.cos(math.radians(carlaTveh[1][1])) + \
                 sinfi * math.sin(math.radians(carlaTveh[1][1]))
        sinsum = math.cos(math.radians(carlaTveh[1][1])) * sinfi - \
                 math.sin(math.radians(carlaTveh[1][1])) * cosfi

        # Append transformed coordinates
        trans_p.append([
            math.sqrt(p[0] ** 2 + p[1] ** 2) * cossum,
            math.sqrt(p[0] ** 2 + p[1] ** 2) * sinsum,
            p[2]
        ])

    # Return transformed camera position, direction, and up vector
    return trans_p[0], \
           [trans_p[1][0] - trans_p[0][0],
            trans_p[1][1] - trans_p[0][1],
            trans_p[1][2] - trans_p[0][2]], \
           [trans_p[2][0] - trans_p[0][0],
            trans_p[2][1] - trans_p[0][1],
            trans_p[2][2] - trans_p[0][2]]


########################################################################
############ Wrapper class for the Neural Renderer ####################
##### All functions must only use numpy arrays as inputs/outputs #######
########################################################################

class NMR(object):
    def __init__(self):
        # Setup renderer
        renderer = neural_renderer.Renderer(camera_mode="projection")
        self.renderer = renderer

    def to_gpu(self, device=0):
        self.cuda_device = device

    def forward_mask(self, vertices, faces):
        ''' Render silhouette masks.
        Args:
            vertices: B x N x 3 numpy array
            faces: B x F x 3 numpy array
        Returns:
            masks: B x 256 x 256 numpy array
        '''
        self.faces = torch.autograd.Variable(faces.cuda())
        self.vertices = torch.autograd.Variable(vertices.cuda())

        self.masks = self.renderer.render_silhouettes(
            self.vertices, self.faces
        )

        masks = self.masks.data.get()
        return masks

    def forward_img(self, vertices, faces, textures):
        ''' Render textured images.
        Args:
            vertices: B x N x 3 numpy array
            faces: B x F x 3 numpy array
            textures: B x F x T x T x T x 3 numpy array
        Returns:
            images: B x 3 x 256 x 256 numpy array
        '''
        self.faces = faces
        self.vertices = vertices
        self.textures = textures
        self.images, _, _ = self.renderer.render(
            self.vertices, self.faces, self.textures
        )
        return self.images


########################################################################
############ Wrapper torch module for Neural Renderer ##################
########################################################################

class NeuralRenderer(torch.nn.Module):
    """
    Core PyTorch renderer interface.
    """

    def __init__(self, img_size=720):
        super(NeuralRenderer, self).__init__()
        self.renderer = NMR()

        # Rendering settings
        self.renderer.renderer.image_size = img_size

        # Camera settings
        self.renderer.renderer.camera_mode = "projection"
        self.renderer.renderer.K = None
        self.renderer.renderer.R = None
        self.renderer.renderer.t = None
        self.renderer.renderer.dist_coeffs = torch.cuda.FloatTensor(
            [[0., 0., 0., 0., 0.]]
        )
        self.renderer.renderer.orig_size = 640

        # Lighting settings
        self.renderer.renderer.light_intensity_ambient = 0.5
        self.renderer.renderer.light_intensity_directional = 0.5
        self.renderer.renderer.light_color_ambient = [1, 1, 1]
        self.renderer.renderer.light_color_directional = [1, 1, 1]
        self.renderer.renderer.light_direction = [0, 0, 1]

        self.renderer.to_gpu()

        self.proj_fn = None
        self.offset_z = 5.

    def ambient_light_only(self):
        # Enable ambient light only
        self.renderer.renderer.light_intensity_ambient = 1
        self.renderer.renderer.light_intensity_directional = 0

    def set_bgcolor(self, color):
        # Set background color
        self.renderer.renderer.background_color = color

    def forward(self, vertices, faces, textures=None):
        if textures is not None:
            return self.renderer.forward_img(vertices, faces, textures)
        else:
            return self.renderer.forward_mask(vertices, faces)


def texture23d():
    obj_file = './src/carassets/audi_et_te.obj'
    img_save_dir = './src/render_res_final/'

    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    texture_size = 6

    # Load model with textures
    vertices, faces, texture = neural_renderer.load_obj(
        obj_file,
        texture_size=texture_size,
        load_texture=True
    )

    texture_origin = texture[None, :, :, :, :, :].cuda(device=0)

    # Create texture mask
    texture_mask = np.zeros(
        (faces.shape[0], texture_size, texture_size,
         texture_size, 3),
        'int8'
    )

    with open('./src/carassets/exterior_face.txt', 'r') as f:
        face_ids = f.readlines()
        for face_id in face_ids:
            texture_mask[int(face_id) - 1, :, :, :, :] = 1

    texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)

    faces_var = torch.autograd.Variable(faces.cuda(device=0))
    vertices_var = vertices.cuda(device=0)

    textures = np.load('./src/textures/texture_camouflage.npy')
    textures = torch.from_numpy(textures).cuda(device=0)

    textures_content = 0.5 * (torch.nn.Tanh()(textures) + 1)

    textures_final = texture_origin * (1 - texture_mask) + \
                     texture_mask * textures_content

    neural_renderer.save_obj(
        img_save_dir + 'final.obj',
        vertices_var,
        faces_var,
        textures_final[0],
        texture_size_out=texture_size
    )