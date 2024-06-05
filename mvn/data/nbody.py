import os

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

DATAROOT = os.environ["DATAROOT"]


class NBodyDataset:
    """
    NBodyDataset

    """

    def __init__(
        self, partition="train", max_samples=1e8, dataset_name="se3_transformer"
    ):
        self.partition = partition
        if self.partition == "val":
            self.sufix = "valid"
        else:
            self.sufix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.sufix += "_charged5_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.sufix += "_charged5_initvel1small"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()

    def load(self):
        # loc = np.load('n_body_system/dataset/loc_' + self.sufix + '.npy')
        loc = np.load(DATAROOT + "/nbody/loc_" + self.sufix + ".npy")
        # vel = np.load('n_body_system/dataset/vel_' + self.sufix + '.npy')
        vel = np.load(DATAROOT + "/nbody/vel_" + self.sufix + ".npy")
        # edges = np.load('n_body_system/dataset/edges_' + self.sufix + '.npy')
        edges = np.load(DATAROOT + "/nbody/edges_" + self.sufix + ".npy")
        # charges = np.load('n_body_system/dataset/charges_' + self.sufix + '.npy')
        charges = np.load(DATAROOT + "/nbody/charges_" + self.sufix + ".npy")

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges

    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc, vel = torch.Tensor(loc).transpose(2, 3), torch.Tensor(vel).transpose(2, 3)
        n_nodes = loc.size(2)
        loc = loc[0 : self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0 : self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0 : self.max_samples]
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        edge_attr = (
            torch.from_numpy(np.array(edge_attr)).float().transpose(0, 1).unsqueeze(2)
        )  # swap n_nodes <--> batch_size and add nf dimension
        return (
            torch.Tensor(loc),
            torch.Tensor(vel),
            torch.Tensor(edge_attr),
            torch.Tensor(edges).long(),
            torch.Tensor(charges),
        )

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)
        # return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T], torch.tensor(i, dtype=torch.long)
        graph_data = Data(
            loc=loc[frame_0],
            vel=vel[frame_0],
            edge_attr=edge_attr,
            charges=charges,
            y=loc[frame_T],
            edge_index=self.edges,
            num_nodes=5
        )

        return graph_data

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges

    
class NBodySimplicialData(InMemoryDataset):
    """NBody Simplicial Dataset."""

    def __init__(
        self,
        root=DATAROOT,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        num_samples=int(1e8),
        partition="train",
    ):
        self.num_samples = num_samples
        self.partition = partition
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"{self.partition}_data.pt"]

    def process(self):
        self.dataset = NBodyDataset(
            partition=self.partition,
            max_samples=self.num_samples,
            dataset_name="nbody_small",
        )
        # Read data into huge `Data` list.
        data_list = [graph for graph in self.dataset]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class NBody:
    def __init__(
        self,
        num_samples=3000,
        batch_size=100,
        simplicial: bool = False,
        dim=2,
        dis=int(1e8),
    ):
        self.simplicial = simplicial
        if not simplicial:
            self.train_dataset = NBodyDataset(
                partition="train", max_samples=num_samples, dataset_name="nbody_small"
            )
            self.valid_dataset = NBodyDataset(
                partition="val", max_samples=num_samples, dataset_name="nbody_small"
            )

            self.test_dataset = NBodyDataset(
                partition="test", max_samples=num_samples, dataset_name="nbody_small"
            )

        else:
            self.transform = SimplicialTransform(dim=dim, dis=dis, label="nbody")
            self.train_dataset = NBodySimplicialData(
                root=f"{DATAROOT}nbody_simplicial_{dim}_{dis}",
                pre_transform=self.transform,
                partition="train",
            )
            self.valid_dataset = NBodySimplicialData(
                root=f"{DATAROOT}nbody_simplicial_{dim}_{dis}",
                pre_transform=self.transform,
                partition="val",
            )
            self.test_dataset = NBodySimplicialData(
                root=f"{DATAROOT}nbody_simplicial_{dim}_{dis}",
                pre_transform=self.transform,
                partition="test",
            )

            self.follow = [f"x_{i}" for i in range(dim + 1)] + ["x", "x_ind"]
        self.batch_size = batch_size

    def train_loader(self):
        if not self.simplicial:
            return PyGDataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            )
        else:
            return PyGDataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                follow_batch=self.follow,
            )

    def val_loader(self):
        if not self.simplicial:
            return PyGDataLoader(
                self.valid_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            return PyGDataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                follow_batch=self.follow,
            )

    def test_loader(self):
        if not self.simplicial:
            return PyGDataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            return PyGDataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                follow_batch=self.follow,
            )
            
