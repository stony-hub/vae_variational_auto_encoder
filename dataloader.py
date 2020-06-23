from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(root, type, img_size, batch_size):
    if type == 'MNIST':
        trans = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.MNIST(root, train=True, transform=trans, download=True)
        test_dataset = datasets.MNIST(root, train=False, transform=trans, download=True)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
        return train_loader, test_loader
    raise Exception('data not implemented')