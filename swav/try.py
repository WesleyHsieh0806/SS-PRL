import torch
from torch.utils.data import Dataset, DataLoader
import torch
import re
from torch._six import container_abcs, string_classes, int_classes


class ImgDataset(Dataset):
    def __init__(self) -> None:
        super(ImgDataset).__init__()

    def __getitem__(self, index: int):
        print(index)
        return [torch.zeros([3, 3, 3])for i in range(6)]

    def __len__(self) -> int:
        return 10


train_dataset = ImgDataset()
Train_loader = DataLoader(train_dataset, batch_size=2)


def checktype(elem):
    batch = [elem]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        print("Tensor")
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        print("numpy")
    elif isinstance(elem, float):
        print("Float")
    elif isinstance(elem, int_classes):
        print("int")
    elif isinstance(elem, string_classes):
        print("string")
    elif isinstance(elem, container_abcs.Mapping):
        print("dict")
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        print("tuuple")
    elif isinstance(elem, container_abcs.Sequence):
        print("sequence")



print(checktype(train_dataset[0]))
