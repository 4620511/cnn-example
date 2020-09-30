from torch.utils.data import DataLoader


def get_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle)
