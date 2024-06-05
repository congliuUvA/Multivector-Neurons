import sidechainnet as scn
import torch
import numpy as np
import torch.utils


def masked_collate_fn(batch):
    max_len = max(len(protein[0]) for protein in batch)

    batch_data = []
    batch_mask = []

    for protein in batch:
        pad_size = max_len - len(protein[0])
        backbone, position, seq = protein

        padded_backbone = np.pad(
            backbone,
            ((0, pad_size), (0, 0), (0, 0)),
            mode="constant",
            constant_values=np.nan,
        )
        padded_position = np.pad(
            position, ((0, pad_size)), mode="constant", constant_values=-1
        )
        padded_seq = np.pad(seq, ((0, pad_size)), mode="constant", constant_values=-1)

        batch_data.append((padded_backbone, padded_position, padded_seq))

        mask = np.concatenate([np.ones(len(backbone)), np.zeros(pad_size)])
        batch_mask.append(mask)

        # padded_x = torch.cat([x, torch.full((pad_size, x.size(1)), torch.nan)])
        # batch_data.append(padded_x)

        # mask = torch.cat([torch.ones(x.size(0)), torch.zeros(pad_size)])
        # batch_mask.append(mask)
    backbones, positions, seqs = zip(*batch_data)
    backbones = np.stack(backbones)
    positions = np.stack(positions)
    seqs = np.stack(seqs)
    mask = np.stack(batch_mask)
    return (
        torch.from_numpy(backbones),
        torch.from_numpy(positions).long(),
        torch.from_numpy(seqs).long(),
        torch.from_numpy(mask).bool(),
    )


class SidechainNetDataset:
    def __init__(self, batch_size=1, max_length=256, min_length=16, train=True):
        self.batch_size = batch_size
        self.max_length = max_length
        self.min_length = min_length
        self.train = train

        self.dataset = scn.load(
            casp_version=12,
            thinning=30,
            batch_size=batch_size,
            dynamic_batching=False,
        )

        self.proteins = []
        self.process()
        if train:
            print("Train dataset size:", len(self))
        else:
            print("Test dataset size:", len(self))

    def process(self):
        for i in range(len(self.dataset)):
            p = self.dataset[i]

            if (p.split == "train") != self.train:
                continue

            coords = p.coords
            backbone = coords[:, :3]
            mask = np.isnan(backbone).any(axis=(1, 2))
            position = np.arange(len(backbone))

            backbone = backbone[~mask]

            if np.isnan(backbone).any():
                continue

            if len(backbone) > self.max_length or len(backbone) < self.min_length:
                continue

            position = position[~mask]
            seq = np.array(p.int_seq)[~mask]

            if np.isnan(position).any() or np.isnan(seq).any():
                continue

            # p.backbone = backbone
            # p.position = position
            # p.seq = seq
            self.proteins.append((backbone, position, seq))

    def __getitem__(self, index):
        return self.proteins[index]

    def __len__(self):
        return len(self.proteins)


class SidechainNet:

    def __init__(self, batch_size=2):
        self.batch_size = batch_size

        self.train = SidechainNetDataset(train=True)
        self.test = SidechainNetDataset(train=False)

    def train_loader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=masked_collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )

    def test_loader(self):
        return torch.utils.data.DataLoader(
            self.test, batch_size=self.batch_size, collate_fn=masked_collate_fn
        )

    def val_loader(self):
        return None


if __name__ == "__main__":

    dataset = SidechainNetDataset()
    dataset_test = SidechainNetDataset(train=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, collate_fn=masked_collate_fn, shuffle=True
    )
    batch = next(iter(dataloader))
    backbone, position, seq, mask = batch
    protein_lengths = mask.sum(dim=1)
    batch_idx = torch.arange(len(backbone)).repeat_interleave(protein_lengths)
