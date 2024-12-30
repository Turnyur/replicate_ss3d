import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import os.path as osp
import random
import time

import imageio
import numpy as np
import scipy.io as sio
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DistributedSampler
from volumetric_render import (
    get_rays_from_angles,
    get_transformation,
)

# Williams added import
import h5py


def loadDepth(dFile, minVal=0, maxVal=10):
    
    dMap = imageio.imread(dFile) # for normal image file

    #dMap = dFile #For hdf5 file
    #dMap = dMap.astype(np.float32)
    
    dMap = dMap * (maxVal - minVal) / (pow(2, 16) - 1) + minVal
    return dMap


def getSynsetsV1Paths(data_cfg):
    synsets = data_cfg.class_ids.split(",")
    synsets.sort()
    root_dir = data_cfg.rgb_dir
    synsetModels = [
        [f for f in os.listdir(osp.join(root_dir, s)) if len(f) > 3] for s in synsets
    ]

    paths = []
    for i in range(len(synsets)):
        for m in synsetModels[i]:
            paths.append([synsets[i], m])

    return paths


def get_rays_multiplex(cameras, rgb_imgs, mask_imgs, render_cfg, device):
    len_cameras = len(cameras)
    assert len_cameras == len(rgb_imgs) and len_cameras == len(
        mask_imgs
    ), "incorrect inputs for camera computation"

    all_rays = []
    all_rgb_labels = []
    all_mask_labels = []
    for i, cam in enumerate(cameras):
        # compute rays

        assert len(cam) == render_cfg.cam_num, "incorrect per frame camera number"
        indices = []
        ind = torch.randperm(render_cfg.img_size * render_cfg.img_size)
        for j in range(len(cam)):
            indices.append(ind[: render_cfg.ray_num_per_cam])
            all_rgb_labels.append(rgb_imgs[i, indices[-1]])
            all_mask_labels.append(mask_imgs[i, indices[-1]])

        all_rays.append(
            get_rays_from_angles(
                H=render_cfg.img_size,
                W=render_cfg.img_size,
                focal=float(render_cfg.focal_length),
                near_plane=render_cfg.near_plane,
                far_plane=render_cfg.far_plane,
                elev_angles=cam[:, 0],
                azim_angles=cam[:, 1],
                dists=cam[:, 2],
                device=device,
                indices=indices,
            )
        )  # [(Num_cams_per_frame*Num_rays), 8] #2d

    return (
        torch.cat(all_mask_labels).to(
            device
        ),  # [(N*Num_cams_per_frame*Num_rays_per_cam)] #1d
        torch.cat(all_rgb_labels).to(
            device
        ),  # [[(N*Num_cams_per_frame*Num_rays_per_cam), 3] #2d
        torch.cat(all_rays).to(
            device
        ),  # [(N*Num_cams_per_frame*Num_rays_per_cam), 8] #2d
    )


def extract_data_train(batch_dict, render_cfg, device):
    # If using pre-rendered.
    assert "mask_label_rays" in batch_dict

    inp_imgs = batch_dict["rgb_img"]
    mask_label_rays = batch_dict["mask_label_rays"].view(-1)
    rays = batch_dict["rays"].view(-1, 8)
    rgb_label_rays = batch_dict["rgb_label_rays"]
    rgb_label_rays = rgb_label_rays.reshape(-1, 3)

    return (
        inp_imgs.to(device),  # [N, 3, img_size, img_size]
        mask_label_rays.to(device),  # [(N*Num_rays)] #1d
        rgb_label_rays.to(device),  # [[(N*Num_rays), 3] #2d
        None,
        None,
        rays.to(device),  # [(N*Num_rays), 8] #2d
    )


class DatasetPermutationWrapper(Dataset):
    def __init__(self, dset):
        self.dset = dset
        self._len = len(self.dset)

    def __len__(self):
        return self._len

    def __getitem__(self, _):
        # TODO(Fix): This random generator behaves same on all gpu's
        index = random.randint(0, self._len - 1)
        return self.dset[index]


class WareHouse3DDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    # TODO: Change hardcoded values!
    def __init__(self, data_root, paths, render_cfg, encoder_root):

        super(WareHouse3DDataset, self).__init__()

        self.paths = paths
        self.render_cfg = render_cfg
        self.data_root = data_root
        self.n_cams = self.render_cfg.cam_num
        self.n_rays_per_cam = self.render_cfg.ray_num_per_cam
        self.encoder_root = encoder_root

        self.transform_img = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.transform_label = transforms.Compose(
            [
                transforms.Resize((self.render_cfg.img_size, self.render_cfg.img_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        st_time = time.time()
        rel_path = os.path.join(self.paths[index][0], self.paths[index][1])
        data_path = os.path.join(self.data_root, rel_path)

        # sample a random encoder imput rgb image.

        encoder_cam_info = np.load(
            osp.join(self.encoder_root, rel_path, "cam_info.npy")
        )
        encoder_sample_num = random.randint(0, encoder_cam_info.shape[0] - 1)
        # encoder_sample_num = 4 # Only for debug
        inp_angles = encoder_cam_info[encoder_sample_num, :]
        inp_angles[0] += 90
        img_path = os.path.join(
            self.encoder_root, rel_path, "render_{}.png".format(encoder_sample_num)
        )
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        inp_rgb_size = img.size[0]
        inp_focal = (sio.loadmat(os.path.join(data_path, "camera_0.mat")))["K"][0, 0]

        img = self.transform_img(img)

        # Sample random cameras
        cam_info = np.load(osp.join(data_path, "cam_info.npy"))

        # Only use a subset of the data
        if self.render_cfg.num_pre_rend_masks > 1:
            cam_info = cam_info[: self.render_cfg.num_pre_rend_masks, :]

        # TODO(Fix): This random generator behaves same on all data-workers on each gpu
        cam_inds = np.random.choice(cam_info.shape[0], self.n_cams)
        # cam_inds = [0] # only for debug
        cam_info = torch.Tensor(cam_info[cam_inds, :])
        azim_angle, elev_angle, theta, dist = torch.split(cam_info, 1, dim=-1)
        azim_angle += (
            90  # This a known blender offset. Look at the noteboosks for visual test.
        )
        # sample rays from cameras and mask images
        render_cfg = self.render_cfg
        pixel_ids = []
        ray_mask_labels = []
        ray_rgb_labels = []

        for i, nc in enumerate(cam_inds):
            temp_idx = torch.randperm(
                self.render_cfg.img_size * self.render_cfg.img_size
            )
            idx = temp_idx[: self.n_rays_per_cam]  # [1, n_rays_per_cam]
            pixel_ids.append(idx)

            # Masks from depth
            gt_mask = loadDepth(
                osp.join(data_path, "depth_{}.png".format(int(nc))), minVal=0, maxVal=10
            )
            empty = gt_mask >= 10.0
            notempty = gt_mask < 10.0
            gt_mask[empty] = 0
            gt_mask[notempty] = 1.0
            gt_mask = self.transform_label(Image.fromarray(gt_mask))
            gt_mask = gt_mask.view(-1).float()
            ray_mask_labels.append(gt_mask[idx])

            # RGB Pixels
            label_rgb_path = osp.join(data_path, "render_{}.png".format(int(nc)))
            with open(label_rgb_path, "rb") as f:
                gt_rgb = Image.open(f)
                gt_rgb = gt_rgb.convert("RGB")

            gt_rgb = self.transform_label(gt_rgb)
            gt_rgb = gt_rgb.permute(1, 2, 0)
            gt_rgb = gt_rgb.reshape(-1, 3)
            ray_rgb_labels.append(gt_rgb[idx])

            # inp_angles = cam_info[i, :]  # TODO: Test!!!

        # n_cams X n_rays_per_cam
        pixel_idx = torch.stack(pixel_ids, dim=0)
        mask_label_rays = torch.cat(ray_mask_labels, dim=0)
        rgb_label_rays = torch.cat(ray_rgb_labels, dim=0)

        label_focal = inp_focal * float(render_cfg.img_size) / inp_rgb_size

        # Used only in relative case
        rays = get_rays_from_angles(
            H=render_cfg.img_size,
            W=render_cfg.img_size,
            focal=label_focal,
            near_plane=render_cfg.near_plane,
            far_plane=render_cfg.far_plane,
            elev_angles=elev_angle[:, 0],
            azim_angles=azim_angle[:, 0],
            dists=dist[:, 0],
            device=torch.device("cpu"),
            indices=pixel_idx,
            transformation_rel=None,
        )  # [(n_cams * n_rays_per_cam), 8] #2d

        return {
            "rgb_img": img,
            "rays": rays,
            "mask_label_rays": mask_label_rays,
            "rgb_label_rays": rgb_label_rays,
            # info useful for debugging
            "elev_angle": torch.tensor([elev_angle[-1, 0]]).float(),
            "azim_angle": torch.tensor([azim_angle[-1, 0]]).float(),
            "dist": torch.tensor([dist[-1, 0]]).float(),
            "rel_path": rel_path,
            # Used for no camera pose
            "label_img_path": label_rgb_path,
            "label_rgb_img": gt_rgb,
            "label_mask_img": gt_mask,
            # class id's
            "class_id": self.paths[index][0],
            "orig_img_path": label_rgb_path,
        }


class WareHouse3DModule(LightningDataModule):
    def __init__(self, data_cfg, render_cfg, num_workers=0, debug_mode=False):
        super().__init__()
        self.data_cfg = data_cfg
        self.render_cfg = render_cfg
        self.num_workers = num_workers

        paths = getSynsetsV1Paths(self.data_cfg)

        train_size = int(self.data_cfg.train_split * len(paths))
        validation_size = int(self.data_cfg.validation_split * len(paths))
        test_size = len(paths) - train_size - validation_size
        (
            self.train_split,
            self.validation_split,
            self.test_split,
        ) = torch.utils.data.random_split(
            paths, [train_size, validation_size, test_size]
        )
        print(
            "Total Number of paths:",
            len(paths),
            len(self.train_split),
            len(self.validation_split),
        )

    def train_dataloader(self):

        assert self.render_cfg.cam_num > 0, "camera number cannot be 0"
        # train_ds = WareHouse3DDataset(
        #     self.data_cfg.rgb_dir,
        #     self.train_split,
        #     self.render_cfg,
        #     self.data_cfg.encoder_dir,
        # )

        train_ds = Shapenet3DCustomDataset(
            self.data_cfg.rgb_dir,
            self.train_split,
            self.render_cfg,
            self.data_cfg.encoder_dir,
        )
        
        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(train_ds)

        return torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.data_cfg.bs_train,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            sampler=sampler,
        )

    def val_dataloader(self):

        assert self.render_cfg.cam_num > 0, "camera number cannot be 0"

        val_ds = DatasetPermutationWrapper(
            # WareHouse3DDataset(
            #     self.data_cfg.rgb_dir,
            #     self.validation_split,
            #     self.render_cfg,
            #     self.data_cfg.encoder_dir,
            # )

            Shapenet3DCustomDataset(
                self.data_cfg.rgb_dir,
                self.validation_split,
                self.render_cfg,
                self.data_cfg.encoder_dir,
            )
        )
        return torch.utils.data.DataLoader(
            val_ds, batch_size=self.data_cfg.bs_val, num_workers=self.num_workers
        )













# Williams added Shapenet3DCustomDataset

class Shapenet3DCustomDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    # TODO: Change hardcoded values!
    def __init__(self, data_root, paths, render_cfg, encoder_root):

        super(Shapenet3DCustomDataset, self).__init__()

        self.paths = paths
        self.render_cfg = render_cfg
        self.data_root = data_root # to comment
        self.n_cams = self.render_cfg.cam_num
        self.n_rays_per_cam = self.render_cfg.ray_num_per_cam
        self.encoder_root = encoder_root # to comment

        #self.h5_file = h5_file

        self.transform_img = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.transform_label = transforms.Compose(
            [
                transforms.Resize((self.render_cfg.img_size, self.render_cfg.img_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        st_time = time.time()

        # ToDo 
        # ['02747177', '9c6176af3ee3918d6140d56bf601ecf2']
        rel_path = os.path.join(self.paths[index][0], self.paths[index][1]) # '03642806/9c6176af3ee3918d6140d56bf601ecf2'

        # '/home/hpc/iwi9/iwi9117h/dev/shapenet_uuid/03642806/9c6176af3ee3918d6140d56bf601ecf2'
        data_path = os.path.join(self.data_root, rel_path) 
        # '/home/hpc/iwi9/iwi9117h/dev/shapenet_uuid/03642806/9c6176af3ee3918d6140d56bf601ecf2'

        rendered_files = os.listdir(data_path) 
        rendered_files.sort()

        file_groups = {'camera': [], 'depth': [], 'render': []}

        for f in rendered_files:
            prefix = f.split('_')[0] # camera_0.mat -> camera
            file_groups[prefix].append(f)




        # sample rays from cameras and mask images
        render_cfg = self.render_cfg
        pixel_ids = []
        ray_mask_labels = []
        ray_rgb_labels = []
        focal_length = 0

        elev_angle = []
        azim_angle = []
        dist = []
        images = []

        # img_path = osp.join(data_path, file_groups['render'][0])  # get first image file
        # with open(img_path, "rb") as f:
        #     img = Image.open(f)
        #     img = img.convert("RGB")
        # inp_rgb_size = img.size[0]
        # image = self.transform_img(img)

        


        for i in range(len(file_groups['camera'])): # iterate over each camera
            temp_idx = torch.randperm(
                self.render_cfg.img_size * self.render_cfg.img_size
            )
            idx = temp_idx[: self.n_rays_per_cam]  # [1, n_rays_per_cam]
            pixel_ids.append(idx)

            # Masks from depth
            gt_mask = loadDepth(
                osp.join(data_path, file_groups['depth'][i]), minVal=0, maxVal=10
            )
            
            empty = gt_mask >= 10.0 # to remove down
            notempty = gt_mask < 10.0 # to remove down
            gt_mask[empty] = 0 # to remove down
            gt_mask[notempty] = 1.0 # to remove down
        
            gt_mask = self.transform_label(Image.fromarray(gt_mask))
            gt_mask = gt_mask.view(-1).float() # flatten
            ray_mask_labels.append(gt_mask[idx])

            # RGB Pixels
            label_rgb_path = osp.join(data_path, file_groups['render'][i])
            with open(label_rgb_path, "rb") as f:
                gt_rgb = Image.open(f)
                gt_rgb = gt_rgb.convert("RGB")

            gt_rgb = self.transform_label(gt_rgb)
            gt_rgb = gt_rgb.permute(1, 2, 0) # to be removed
            gt_rgb = gt_rgb.reshape(-1, 3) # flatten and keep channels intact #3.
            ray_rgb_labels.append(gt_rgb[idx])

            cam_path = osp.join(data_path, file_groups['camera'][i])
            camera_matrix = sio.loadmat(cam_path)
            cam_index = "camera_{}".format(int(i))
            camera_matrix = camera_matrix[cam_index]
            #camera_data = camera_matrix[0, 0]

            K = camera_matrix[0]
            RT = camera_matrix[1]


            azimuthal_angle_rad = np.arctan2(RT[1, 2], RT[0, 2])
            elevation_angle_rad = -np.arcsin(RT[2, 2])

            azimuthal_angle = np.degrees(azimuthal_angle_rad)
            elevation_angle = np.degrees(elevation_angle_rad)

            azim_angle.append(azimuthal_angle)
            elev_angle.append(elevation_angle)

            translation_vector = RT[:3, 3]
            # Calculate the distance using the Euclidean norm of the translation vector
            distance = np.linalg.norm(translation_vector)
            dist.append(distance)
            
            f_len = K[0,0]
            focal_length = f_len

            img_path = osp.join(data_path, file_groups['render'][0])  # get first image file
            with open(img_path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")
            inp_rgb_size = img.size[0]
            image = self.transform_img(img)
            images.append(image)

            # inp_angles = cam_info[i, :]  # TODO: Test!!!

        # n_cams X n_rays_per_cam
        pixel_idx = torch.stack(pixel_ids, dim=0)
        mask_label_rays = torch.cat(ray_mask_labels, dim=0)
        rgb_label_rays = torch.cat(ray_rgb_labels, dim=0)

        label_focal = focal_length #* float(render_cfg.img_size) / inp_rgb_size
        elev_angle_array = torch.tensor(np.array(elev_angle).reshape(-1, 1), dtype=torch.float)
        azim_angle_array = torch.tensor(np.array(azim_angle).reshape(-1, 1), dtype=torch.float)
        dist_array = torch.tensor(np.array(dist).reshape(-1, 1), dtype=torch.float)

        # Used only in relative case
        rays = get_rays_from_angles(
            H=render_cfg.img_size,
            W=render_cfg.img_size,
            focal=label_focal,
            near_plane=render_cfg.near_plane,
            far_plane=render_cfg.far_plane,
            elev_angles=elev_angle_array[:, 0],
            azim_angles=azim_angle_array[:, 0],
            dists=dist_array[:, 0],
            device=torch.device("cpu"),
            indices=pixel_idx,
            transformation_rel=None,
        )  # [(n_cams * n_rays_per_cam), 8] #2d

        return {
            "rgb_img": torch.stack(images)[0],
            "rays": rays,
            "mask_label_rays": mask_label_rays,
            "rgb_label_rays": rgb_label_rays,
            # info useful for debugging
            "elev_angle": elev_angle_array[:, 0],
            "azim_angle": azim_angle_array[:, 0],
            "dist": dist_array[:, 0],
            "rel_path": rel_path,
            # Used for no camera pose
            "label_img_path": label_rgb_path,
            "label_rgb_img": gt_rgb,
            "label_mask_img": gt_mask,
            # class id's
            "class_id": self.paths[index][0],
            "orig_img_path": label_rgb_path,
        }








# H5 Adopted Williams
class ShapenetH5Dataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    # TODO: Change hardcoded values!
    def __init__(self, h5_handle, paths, render_cfg):

        super(ShapenetH5Dataset, self).__init__()

        self.shapenet_h5 = h5_handle
        self.paths = paths
        self.render_cfg = render_cfg
        self.n_cams = self.render_cfg.cam_num
        self.n_rays_per_cam = self.render_cfg.ray_num_per_cam
        

        self.transform_img = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.transform_label = transforms.Compose(
            [
                transforms.Resize((self.render_cfg.img_size, self.render_cfg.img_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        st_time = time.time()

        # ['02747177', '9c6176af3ee3918d6140d56bf601ecf2']
        data_path = '' # Remove
        rel_path = '' # Remove
        label_rgb_path = ''

        #rendered_files = os.listdir(data_path) 
        obj_data = self.paths[index]
        synset = obj_data[0]
        uuid = obj_data[1]
        rendered_files = self.shapenet_h5[synset][uuid] # Get rendered files'
        #rendered_files.sort()

        file_groups = {'camera': [], 'depth': [], 'render': []}

        for key in rendered_files.keys():
            prefix = key.split('_')[0] # camera_0.mat -> camera
            file_groups[prefix].append(key)




        # sample rays from cameras and mask images
        render_cfg = self.render_cfg
        pixel_ids = []
        ray_mask_labels = []
        ray_rgb_labels = []
        focal_length = 0

        elev_angle = []
        azim_angle = []
        dist = []
        images = []

        # img_path = osp.join(data_path, file_groups['render'][0])  # get first image file
        # with open(img_path, "rb") as f:
        #     img = Image.open(f)
        #     img = img.convert("RGB")
        # inp_rgb_size = img.size[0]
        # image = self.transform_img(img)

        


        for i in range(len(file_groups['camera'])): # iterate over each camera
            temp_idx = torch.randperm(
                self.render_cfg.img_size * self.render_cfg.img_size
            )
            idx = temp_idx[: self.n_rays_per_cam]  # [1, n_rays_per_cam]
            pixel_ids.append(idx)

            # Masks from depth
            gt_mask = loadDepth(
               rendered_files[file_groups['depth'][i]][()], minVal=0, maxVal=10
            )
            
            empty = gt_mask >= 10.0 # to remove down
            notempty = gt_mask < 10.0 # to remove down
            gt_mask[empty] = 0 # to remove down
            gt_mask[notempty] = 1.0 # to remove down
        
            gt_mask = self.transform_label(Image.fromarray(gt_mask))
            gt_mask = gt_mask.view(-1).float() # flatten
            ray_mask_labels.append(gt_mask[idx])

            # RGB Pixels
            gt_rgb = rendered_files[file_groups['render'][i]][()]

            gt_rgb = self.transform_label(Image.fromarray(gt_rgb))
            gt_rgb = gt_rgb.permute(1, 2, 0) # to be removed
            gt_rgb = gt_rgb.reshape(-1, 3) # flatten and keep channels intact #3.
            ray_rgb_labels.append(gt_rgb[idx])

            camera_matrix = rendered_files[file_groups['camera'][i]][()]

            K = camera_matrix[0]
            RT = camera_matrix[1]


            azimuthal_angle_rad = np.arctan2(RT[1, 2], RT[0, 2])
            elevation_angle_rad = -np.arcsin(RT[2, 2])

            azimuthal_angle = np.degrees(azimuthal_angle_rad)
            elevation_angle = np.degrees(elevation_angle_rad)

            azim_angle.append(azimuthal_angle)
            elev_angle.append(elevation_angle)

            translation_vector = RT[:3, 3]
            # Calculate the distance using the Euclidean norm of the translation vector
            distance = np.linalg.norm(translation_vector)
            dist.append(distance)
            
            f_len = K[0,0]
            focal_length = f_len

            img = rendered_files[file_groups['render'][0]][()]  # get first image file
            img = Image.fromarray(img) # get first image file
            image = self.transform_img(img)
            #inp_rgb_size = img.size[image]
            images.append(image)

            # inp_angles = cam_info[i, :]  # TODO: Test!!!

        # n_cams X n_rays_per_cam
        pixel_idx = torch.stack(pixel_ids, dim=0)
        mask_label_rays = torch.cat(ray_mask_labels, dim=0)
        rgb_label_rays = torch.cat(ray_rgb_labels, dim=0)

        label_focal = focal_length #* float(render_cfg.img_size) / inp_rgb_size
        elev_angle_array = torch.tensor(np.array(elev_angle).reshape(-1, 1), dtype=torch.float)
        azim_angle_array = torch.tensor(np.array(azim_angle).reshape(-1, 1), dtype=torch.float)
        dist_array = torch.tensor(np.array(dist).reshape(-1, 1), dtype=torch.float)

        # Used only in relative case
        rays = get_rays_from_angles(
            H=render_cfg.img_size,
            W=render_cfg.img_size,
            focal=label_focal,
            near_plane=render_cfg.near_plane,
            far_plane=render_cfg.far_plane,
            elev_angles=elev_angle_array[:, 0],
            azim_angles=azim_angle_array[:, 0],
            dists=dist_array[:, 0],
            device=torch.device("cpu"),
            indices=pixel_idx,
            transformation_rel=None,
        )  # [(n_cams * n_rays_per_cam), 8] #2d

        return {
            "rgb_img": torch.stack(images)[0],
            "rays": rays,
            "mask_label_rays": mask_label_rays,
            "rgb_label_rays": rgb_label_rays,
            # info useful for debugging
            "elev_angle": elev_angle_array[:, 0],
            "azim_angle": azim_angle_array[:, 0],
            "dist": dist_array[:, 0],
            "rel_path": rel_path,
            # Used for no camera pose
            "label_img_path": label_rgb_path,
            "label_rgb_img": gt_rgb,
            "label_mask_img": gt_mask,
            # class id's
            "class_id": self.paths[index][0],
            "orig_img_path": label_rgb_path,
        }




# Williams added Shapenet3DCustomModule
class Shapenet3DCustomModule(LightningDataModule):
    def __init__(self, h5_file, data_cfg, render_cfg, num_workers=0, debug_mode=False):
        super().__init__()
        self.data_cfg = data_cfg
        self.render_cfg = render_cfg
        self.num_workers = num_workers

        
        # custom H5 loader
        #self.shapenet_h5 = h5_file
        #self.file_handle = h5py.File(self.shapenet_h5, 'r')
        self.file_handle = h5py.File(h5_file, 'r')

        #paths = getSynsetsV1Paths(self.data_cfg)
        synsets = list(self.file_handle.keys()) #data_cfg.class_ids.split(",") 
        synsets.sort()
        root_dir = data_cfg.rgb_dir
        #synsetModels = [
        #    [f for f in os.listdir(osp.join(root_dir, s)) if len(f) > 3] for s in synsets
        #]
        synsetModels = [
            [f for f in self.file_handle[s] ] for s in synsets
        ]

        paths = []
        for i in range(len(synsets)):
            for m in synsetModels[i]:
                paths.append([synsets[i], m])
        # paths end


        train_size = int(self.data_cfg.train_split * len(paths))
        validation_size = int(self.data_cfg.validation_split * len(paths))
        test_size = len(paths) - train_size - validation_size
        (
            self.train_split,
            self.validation_split,
            self.test_split,
        ) = torch.utils.data.random_split(
            paths, [train_size, validation_size, test_size]
        )
        print(
            "Total Number of paths:",
            len(paths),
            len(self.train_split),
            len(self.validation_split),
        )

    def train_dataloader(self):

        assert self.render_cfg.cam_num > 0, "camera number cannot be 0"
        # train_ds = WareHouse3DDataset(
        #     self.data_cfg.rgb_dir,
        #     self.train_split,
        #     self.render_cfg,
        #     self.data_cfg.encoder_dir,
        # )

        train_ds = ShapenetH5Dataset(
            self.file_handle,
            self.train_split,
            self.render_cfg,
        )

        sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sampler = DistributedSampler(train_ds)

        return torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.data_cfg.bs_train,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            sampler=sampler,
        )

    def val_dataloader(self):

        assert self.render_cfg.cam_num > 0, "camera number cannot be 0"

        val_ds = DatasetPermutationWrapper(
            # WareHouse3DDataset(
            #     self.data_cfg.rgb_dir,
            #     self.validation_split,
            #     self.render_cfg,
            #     self.data_cfg.encoder_dir,
            # )

            ShapenetH5Dataset(
                self.file_handle,
                self.train_split,
                self.render_cfg,
            )
        )
        return torch.utils.data.DataLoader(
            val_ds, batch_size=self.data_cfg.bs_val, num_workers=self.num_workers
        )
