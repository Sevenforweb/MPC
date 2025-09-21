from collections import OrderedDict
from functools import lru_cache
import numpy as np
import tqdm
import os
import json
import pickle
import open3d as o3d

from torch.utils.data import Dataset
from pathlib import Path
from utils import box_np_ops

from models.backbones.pointcept.datasets.builder import DATASETS
from models.backbones.pointcept.datasets.transform import Compose
import smplx
def bbox_from_joints(joints_3d: np.ndarray,
                     raidus_scale: float = 1.25):
    cmax = joints_3d.max(axis=-2)
    cmin = joints_3d.min(axis=-2)
    center = (cmin + cmax) / 2
    return np.concatenate([center, 
                           (cmax - cmin) * raidus_scale,
                           np.zeros((center.shape[0], 1))], axis=-1)

@DATASETS.register_module()
class HumanM3Dataset(Dataset):
    def __init__(self,
                 raw_path='datasets/Human_M3',
                 buffer_path='data/human-m3',
                 split='train',
                 smpl_model_path='./smpl_models',
                 transforms=None,
                 keypoint_range=[*range(15)],
                 interval=1,
                 ):
        
        self.raw_path = Path(raw_path)
        self.buffer_path = Path(buffer_path) if buffer_path else Path("./data")
        self.split = split
        self.keypoint_range = [*range(15)]
        self.interval = interval
        self.human_model = smplx.create('./smpl_models/', model_type = 'smpl',
                            gender='neutral', 
                            use_face_contour=False,
                            ext="npz").cuda()
        self.smpl_model_path = Path(smpl_model_path)
        part_mapping = json.load(open(self.smpl_model_path / 'smpl_body_parts_2_faces.json'))
        faces = self.human_model.faces
        faces_label = np.zeros(faces.shape[0], dtype=np.int32)
        for i, val in enumerate(part_mapping.values()):
            faces_label[val] = i + 1
        self.smpl_mixin = dict(
            face=faces,
            face_label=faces_label
        )#有关face都是新增
        if not self.buffer_path.exists():
            os.makedirs(self.buffer_path)
        self._db = self._get_db()
        self.transforms = Compose(transforms)
        
    def __len__(self):
        return len(self._db) // self.interval
    
    @lru_cache(maxsize=16)
    def _lazy_load_pcd(self, pcd_file: str):
        return np.asarray(o3d.io.read_point_cloud(pcd_file).points)
    
    def _get_sensor_data(self, instance_info, load=True):
        if load:
            pcd_file = instance_info['pcd']
            return dict(point_cloud=self._lazy_load_pcd(pcd_file))
        else:
            return dict(point_cloud=np.zeros((1, 3)))
    
    def _get_db(self):
        db_file = self.buffer_path / f"{self.split}.pkl"
        if db_file.exists():
            ord_db = pickle.load(open(db_file, 'rb'))
            new_db = []
            for frame in ord_db['db']:
                pcd_file = frame['pcd']
                for camera_frame in frame['info']:
                    bboxs = bbox_from_joints(camera_frame['joints_3d'])
                    joints_3d = np.concatenate([camera_frame['joints_3d'], 
                                                (camera_frame['joints_3d_vis'] > 0).all(axis=-1, keepdims=True)], axis=-1)
                    for i in range(bboxs.shape[0]):
                        new_db.append({
                            'pcd': pcd_file,
                            'bbox': bboxs[i],
                            'keypoints_3d': joints_3d[i],
                            'keypoints_2d': camera_frame['joints_2d'][i],
                            'camera': camera_frame['camera'],
                        })
            return new_db
        else:
            raise NotImplementedError()
        return []    
    
    def __getitem__(self, idx):
        sample = dict(**self._db[idx * self.interval],**self.smpl_mixin)
        instance_info = self._db[idx * self.interval]
        if 'beta' not in instance_info:
            print(f"Missing 'beta' in sample at idx {idx * self.interval}: keys={list(instance_info.keys())}")
        sensor_data = self._get_sensor_data(instance_info)
        point_cloud = sensor_data['point_cloud']
        sample['coord'] = point_cloud.astype(np.float32)
        bbox = instance_info['bbox']
        if sample['coord'].shape[0] < 10:
            return self.__getitem__(idx + 1)
        # # bypass the false-annotated instance
        # if inst_point_cloud.shape[0] == 0:
        #     return self.__getitem__(idx + 1)
        
        # sample = {
        #     **instance_info,
        #     'coord': inst_point_cloud,
        # }
        # sample['vertex'] = sample['vertices']
        smpl_vertices = self._generate_vertex_data(instance_info)
        sample['vertex'] = smpl_vertices
        sample['keypoints_3d'] = sample['keypoints_3d'][self.keypoint_range].copy()
        return self.transforms(sample)
    def _generate_vertex_data(self, sample):
        import torch
        # 使用 SMPL 模型生成顶点数据
        smpl_params = {
            'betas': torch.tensor(sample['beta']).cuda(),
            'body_pose': torch.tensor(sample['pose'][3:]).cuda(),
            'global_orient': torch.tensor(sample['pose'][:3]).cuda(),
            'transl': torch.tensor(sample['trans']).cuda()
        }
        smpl_output = self.human_model(**smpl_params)
        vertices = smpl_output.vertices.detach().cpu().numpy()[0]
        return vertices