from torchvision.transforms import ToTensor, Lambda
import CustomSegmentationDataset

custom_dataset = CustomSegmentationDataset(root_dir='path_to_root_directory', transform=transform, subsample_factor=2)

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(3, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),
    subsample_factor=2
)

batch_size = 32
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)


from torchvision.transforms import ToTensor, Lambda
import CustomSegmentationDataset

custom_dataset = CustomSegmentationDataset(root_dir='path_to_root_directory', transform=transform, subsample_factor=2)

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(3, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)),
    subsample_factor=2
)

batch_size = 32
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)