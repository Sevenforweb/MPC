import os
import numpy as np
import pickle

from pathlib import Path
from torch.utils.data import Dataset

from models.backbones.pointcept.datasets.builder import DATASETS
from models.backbones.pointcept.datasets.transform import Compose

@DATASETS.register_module()
class WaymoPoseDataset(Dataset):
    def __init__(self,
                 raw_path='./datasets/waymo_v2/',
                 buffer_path='./data/waymo_v2',
                 split='train',
                 modality=['lidar',],
                 keypoint_range=[*range(1, 14)],
                 transforms=None,
                 ) -> None:
        super().__init__()
        self.raw_path = Path(raw_path) / split
        self.buffer_path = Path(buffer_path)
        if not self.buffer_path.exists():
            os.makedirs(self.buffer_path)
        
        self.split = split
        self.keypoint_range = keypoint_range
        self.modality = modality

        # 调试信息
        print(f"Initializing WaymoPoseDataset with split: {self.split}")

        self._db = self._read_infos()
        self._lidar_buffers = dict()
        self._lidar_buffers_fresh = dict()
        
        # 动态调整 keypoint_range
        max_keypoints = self._db[0]['keypoints_3d'].shape[0]
        self.keypoint_range = [k for k in self.keypoint_range if k < max_keypoints]

        self.transforms = Compose(transforms)

    def _read_infos(self):
        db_buffer_file = self.buffer_path / f"{self.split}.pkl"
        # 检查 self.buffer_path 是否存在
        if not self.buffer_path.exists():
            raise FileNotFoundError(f"Buffer path {self.buffer_path} does not exist.")
        if not self.split in ['train', 'test']:
            print(f"{self.split} ")
            raise ValueError(f"Invalid split {self.split}")
        
        if db_buffer_file.exists():
            db = pickle.load(open(db_buffer_file, 'rb'))
            # 调试信息：打印数据库的基本信息
            #print(f"Loaded database from {db_buffer_file}")
            #print(f"Number of samples: {len(db)}")
            #print(f"First sample keys: {list(db[0].keys())}")
            #print(f"First sample keypoints_3d shape: {db[0]['keypoints_3d'].shape}")
            return db
        else:
            raise Exception("Please generate database first.")

    def __getitem__(self, index):
        instance_info = self._db[index]
        sample = dict(**instance_info)
        sample['id'] = index
        sample['coord'], sample['color'] = sample['coord'][:, :3], sample['coord'][:, 3:]
        
        # 调试信息：检查 keypoints_3d 的形状和 keypoint_range
        assert 'keypoints_3d' in sample, f"Missing 'keypoints_3d' in sample {index}"
        assert sample['keypoints_3d'].shape[0] > max(self.keypoint_range), (
            f"keypoint_range {self.keypoint_range} exceeds keypoints_3d shape {sample['keypoints_3d'].shape}"
        )
        
        sample['keypoints_3d'] = sample['keypoints_3d'].copy()[self.keypoint_range, :]
        sample['keypoints_3d'][..., 3] = (sample['keypoints_3d'][..., 3] > 0).astype(np.float32)


        transformed_sample = self.transforms(sample)
    
        return self.transforms(sample)
    
    def __len__(self):
        return len(self._db)