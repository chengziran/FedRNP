import os
import re
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, ConcatDataset


class ACDCDataset(Dataset):
    def __init__(self, data_root, target_size=(256, 256)):
        self.root_dir = data_root
        self.target_size = target_size

        image_dir, label_dir = self._resolve_dirs(data_root)

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image dir not found: {image_dir}")

        if not os.path.isdir(label_dir):
            raise FileNotFoundError(f"Label dir not found: {label_dir}")

        image_dict = self._collect_files(image_dir)
        label_dict = self._collect_files(label_dir)

        common_keys = sorted(set(image_dict.keys()) & set(label_dict.keys()))

        if len(common_keys) == 0:
            raise RuntimeError(
                f"\n[ACDCDataset] No matched image-label pairs found.\n"
                f"image_dir: {image_dir}\n"
                f"label_dir: {label_dir}\n"
                f"num_images: {len(image_dict)}\n"
                f"num_labels: {len(label_dict)}\n"
            )

        self.image_paths = [image_dict[k] for k in common_keys]
        self.label_paths = [label_dict[k] for k in common_keys]

        assert len(self.image_paths) == len(self.label_paths), (
            f"图像数量和标签数量不一致！"
            f"{len(self.image_paths)} vs {len(self.label_paths)}"
        )

    def _resolve_dirs(self, data_root):
        norm_root = os.path.normpath(data_root).replace("\\", "/")
        base_name = os.path.basename(norm_root)
        parent_dir = os.path.dirname(norm_root)

        if base_name == "data":
            image_dir = norm_root

            label_candidates = [
                os.path.join(parent_dir, "label"),
                os.path.join(parent_dir, "labels"),
            ]

            for label_dir in label_candidates:
                if os.path.isdir(label_dir):
                    return image_dir, label_dir

            raise FileNotFoundError(
                f"检测到 data 目录，但未找到对应 label 目录。\n"
                f"data_root: {data_root}\n"
                f"尝试过: {label_candidates}"
            )

        match = re.search(r"train_data_?(\d+)$", norm_root)

        if match:
            folder_id = match.group(1)
            image_dir = norm_root

            label_candidates = [
                os.path.join(parent_dir, f"train_label_{folder_id}"),
                os.path.join(parent_dir, f"train_label{folder_id}"),
                os.path.join(parent_dir, f"label_{folder_id}"),
                os.path.join(parent_dir, f"label{folder_id}"),
            ]

            for label_dir in label_candidates:
                if os.path.isdir(label_dir):
                    return image_dir, label_dir

            raise FileNotFoundError(
                f"检测到 train_data 客户端目录，但未找到对应标签目录。\n"
                f"data_root: {data_root}\n"
                f"尝试过: {label_candidates}"
            )

        raise ValueError(
            f"不支持的 data_root 路径格式: {data_root}\n"
            f"当前支持:\n"
            f"  1) xxx/train_data_0\n"
            f"  2) xxx/data"
        )

    def _collect_files(self, folder):
        valid_exts = [".npy", ".png"]
        file_dict = {}

        for f in os.listdir(folder):
            full_path = os.path.join(folder, f)

            if not os.path.isfile(full_path):
                continue

            stem, ext = os.path.splitext(f)
            ext = ext.lower()

            if ext in valid_exts:
                file_dict[stem] = full_path

        return file_dict

    def _load_array(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext == ".npy":
            arr = np.load(path)

        elif ext == ".png":
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if arr is None:
                raise RuntimeError(f"读取 PNG 失败: {path}")

            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        else:
            raise ValueError(f"不支持的文件格式: {path}")

        if arr.ndim == 3:
            arr = np.squeeze(arr)

        if arr.ndim != 2:
            raise ValueError(
                f"期望 2D 数组，但得到 shape={arr.shape}, path={path}"
            )

        return arr

    def _resize_image(self, img):
        if self.target_size is None:
            return img

        th, tw = self.target_size

        if img.shape != (th, tw):
            img = cv2.resize(
                img,
                (tw, th),
                interpolation=cv2.INTER_LINEAR
            )

        return img

    def _resize_label(self, label):
        if self.target_size is None:
            return label

        th, tw = self.target_size

        if label.shape != (th, tw):
            label = cv2.resize(
                label,
                (tw, th),
                interpolation=cv2.INTER_NEAREST
            )

        return label

    def _convert_label_to_onehot(self, label):
        label = np.asarray(label)
        unique_vals = np.unique(label)

        if set(unique_vals.tolist()).issubset({0, 1, 2, 3}):
            label_cls = label.astype(np.int64)

        elif set(unique_vals.tolist()).issubset({0, 85, 170, 255}):
            label_cls = np.zeros_like(label, dtype=np.int64)
            label_cls[label == 85] = 1
            label_cls[label == 170] = 2
            label_cls[label == 255] = 3

        else:
            vals = sorted(unique_vals.tolist())
            label_cls = np.zeros_like(label, dtype=np.int64)

            for new_id, old_val in enumerate(vals[:4]):
                label_cls[label == old_val] = new_id

        label_cls = torch.tensor(label_cls, dtype=torch.long)

        onehot = torch.zeros(
            4,
            label_cls.shape[0],
            label_cls.shape[1],
            dtype=torch.float32
        )

        onehot.scatter_(0, label_cls.unsqueeze(0), 1.0)

        return onehot

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        img = self._load_array(img_path)
        label = self._load_array(label_path)

        img = self._resize_image(img)
        label = self._resize_label(label)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        label = self._convert_label_to_onehot(label)

        return img, label


def get_train_data_loader(
    batch_size,
    data_root,
    target_size=(256, 256),
    shuffle=True,
    num_workers=0
):
    if isinstance(data_root, str):
        train_data = ACDCDataset(
            data_root,
            target_size=target_size
        )

    elif isinstance(data_root, (list, tuple)):
        datasets = [
            ACDCDataset(
                p,
                target_size=target_size
            )
            for p in data_root
        ]

        train_data = ConcatDataset(datasets)

    else:
        raise TypeError(
            f"Unsupported data_root type: {type(data_root)}"
        )

    if len(train_data) == 0:
        raise RuntimeError(f"Empty dataset: {data_root}")

    dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )

    return dataloader


class VAEDataset(Dataset):
    def __init__(self, root_dir, target_size=(256, 256)):
        self.root_dir = root_dir
        self.target_size = target_size

        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(".npy")
        ])

    def __len__(self):
        return len(self.image_files)

    def _resize_image(self, img):
        if self.target_size is None:
            return img

        th, tw = self.target_size

        if img.shape != (th, tw):
            img = cv2.resize(
                img,
                (tw, th),
                interpolation=cv2.INTER_LINEAR
            )

        return img

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir,
            self.image_files[idx]
        )

        img = np.load(img_name)

        if img.ndim == 3:
            img = np.squeeze(img)

        if img.ndim != 2:
            raise ValueError(
                f"期望 2D 图像，但得到 shape={img.shape}, "
                f"path={img_name}"
            )

        img = self._resize_image(img)

        img = torch.tensor(
            img,
            dtype=torch.float32
        ).unsqueeze(0)

        return img


def get_generate_data_loader(
    batch_size,
    save_path,
    target_size=(256, 256),
    num_workers=0
):
    train_data = VAEDataset(
        save_path,
        target_size=target_size
    )

    dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True
    )

    return dataloader
