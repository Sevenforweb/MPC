import os
import argparse

import pickle
import torch
import smplx
import math
import numpy as np
import open3d as o3d
import tqdm
import json
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict

from models.backbones.pointcept.datasets.builder import DATASETS
from models.backbones.pointcept.datasets.transform import Compose

@DATASETS.register_module()
class SLOPER4DDataset(Dataset):
    def __init__(self, 
                 raw_path="/hy-tmp/dapt-main/datasets/sloper4d/xixihahaya___SLOPER4D-dataset",#原"hy-tmp/dapt-main/datasets/sloper4d"
                 buffer_path="./data/sloper4d",
                 smpl_model_path='./smpl_models',
                 split='train',
                 keypoint_range=[0,1,2,4,5,7,8,12,15,16,17,18,19,20,21],
                 interval=1,
                 transforms=None,
                 modality=None,  # 添加 modality 参数
                 ):
        self.raw_path = Path(raw_path)
        self.buffer_path = Path(buffer_path)
        self.smpl_model_path = Path(smpl_model_path)
        self.split = split
        self.interval = interval
        self.keypoint_range = keypoint_range
        self.modality = modality  # 保存 modality 参数
        self.human_model = smplx.create('./smpl_models/', model_type = 'smpl',
                            gender='neutral', 
                            use_face_contour=False,
                            ext="npz").cuda()
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

        self._db = self._get_info()
        self.transforms = Compose(transforms)
    
    def _get_info(self):
        buffer_file = self.buffer_path / f"{self.split}.pkl"
        
        if buffer_file.exists():
            return pickle.load(open(buffer_file, 'rb'))
        sequences = sorted(self.raw_path.glob("*/*_labels.pkl"))
        tmp_db = defaultdict(list)
        for f in sequences:
            data = pickle.load(open(f, 'rb'))
            human_infos = data['second_person']
            # 调试信息
            if 'pose' not in human_infos:
                raise KeyError(f"Missing key 'pose' in human_infos for file {f}")

            frame_num = np.array(data['frame_num'])
            mask = np.in1d(frame_num, human_infos['point_frame'])
            seqence_len = mask.sum()
            if self.split == 'train':
                patch_size = 16
                patch_len = math.ceil(seqence_len / patch_size)             
                selection = np.zeros(patch_len).astype(bool)
                selection[np.random.permutation(patch_len)[:int(patch_len * 0.7)]] = True
                selection = selection.repeat(patch_size)[:seqence_len]
                (~selection).tofile(self.buffer_path / f.name.replace('_labels.pkl', '_val_sel.bin'))
            else:
                selection = np.fromfile(self.buffer_path / f.name.replace('_labels.pkl', '_val_sel.bin'), dtype=bool)
            seqence_len = selection.sum()
            tmp_db['frame_id'].append(frame_num[mask][selection])
            tmp_db['seqence_id'].append(np.array([data['seqence_name']] * seqence_len))
            tmp_db['gender'].append(np.array([human_infos['gender']] * seqence_len))
            tmp_db['beta'].append(np.array(human_infos['beta'])[None].repeat(seqence_len, axis=0))
            tmp_db['pose'].append(human_infos['pose'][mask][selection])
            tmp_db['trans'].append(human_infos['trans'][mask][selection])
            tmp_db['point_clouds'].append(np.array(human_infos['point_clouds'])[selection])
        tmp_db = {k: np.concatenate(v, axis=0) for k, v in tmp_db.items()}
        tmp_db.update(self._smpl_extract_joint(tmp_db))#human_models在这里传给tmp_db

        # 调试信息
        required_keys = ['pose', 'beta', 'trans']
        for key in required_keys:
            if key not in tmp_db:
                raise KeyError(f"Missing key '{key}' in tmp_db")

        db = []
        keys = [*tmp_db.keys()]
        for i in range(tmp_db['seqence_id'].shape[0]):
            db.append({
                k: tmp_db[k][i] for k in keys
            })

        pickle.dump(db, open(buffer_file, 'wb'))
        return db

    def _smpl_extract_joint(self, 
                            db,
                            batch_size=64,
                            out_keys=['joints', 'vertices']
                            ):
        # 检查 db 是否包含必要的键
        required_keys = ['pose', 'beta', 'trans']
        for key in required_keys:
            if key not in db:
                raise KeyError(f"Missing key '{key}' in db")

        def sample_gen():
            poses, betas, trans = db['pose'], db['beta'], db['trans']
            for i in range(0, poses.shape[0], batch_size):
                yield poses[i:i+batch_size], betas[i:i+batch_size], trans[i:i+batch_size]

        results = {k: [] for k in out_keys}

        
        with torch.no_grad():
            for b_pose, b_beta, b_trans in tqdm.tqdm(sample_gen(), total=db['pose'].shape[0] // batch_size + 1):
                b_pose = Tensor(b_pose).cuda().float()
                b_beta = Tensor(b_beta).cuda().float()
                b_trans = Tensor(b_trans).cuda().float()
                
                smpl_out = self.human_model(
                    betas=b_beta,
                    body_pose=b_pose[..., 3:],
                    global_orient=b_pose[..., :3],
                    transl=b_trans
                )
                for k in out_keys:
                    results[k].append(smpl_out[k].cpu().numpy())
            
        return {k: np.concatenate(results[k], axis=0) for k in out_keys}

    def __getitem__(self, idx):
        sample = dict(**self._db[idx * self.interval],**self.smpl_mixin)#db是上面那个,新增face
        sample['id'] = idx
        sample['frame_id'] = sample['frame_id'].item()
        sample['coord'] = sample['point_clouds'].astype(np.float32)
        if sample['coord'].shape[0] < 10:
            return self.__getitem__(idx + 1)
        sample['bbox'] = sample['trans'].astype(np.float32)
        sample['keypoints_3d'] = sample['joints'][self.keypoint_range].copy()
        sample['keypoints_3d'] = np.concatenate([sample['keypoints_3d'], 
                                                 np.ones((sample['keypoints_3d'].shape[0], 1)).astype(np.float32)], axis=1)
        
        #new
        sample['vertex'] = sample['vertices']
        
        # ----------- 兼容 LPFormer 字段名（后续如果要跑其他模型记得要删除） -----------
        # 不要拼接置信度
        sample['smpl_joints_local'] = sample['keypoints_3d']  # shape [K, 3]
        sample['seg_label'] = np.zeros(sample['coord'].shape[0], dtype=np.int64)  # 没有分割标签时全0
        sample['vis_label'] = np.ones(sample['coord'].shape[0], dtype=np.int64)   # 没有可见性标签时全1
        # -----------------------------------
        return self.transforms(sample)
    
    def __len__(self):
        return len(self._db) // self.interval
