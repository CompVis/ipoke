from data.base_dataset import BaseDataset
from torchvision import transforms as tt
from data.flow_dataset import PlantDataset, IperDataset,Human36mDataset,  TaichiDataset
from data.samplers import SequenceSampler,FixedLengthSampler,SequenceLengthSampler

# add key value pair for datasets here, all datasets should inherit from base_dataset
__datasets__ = {"IperDataset": IperDataset,
                "PlantDataset": PlantDataset,
                "Human36mDataset": Human36mDataset,
                "TaichiDataset": TaichiDataset,}

__samplers__ = {"fixed_length": FixedLengthSampler,
                }


# returns only the class, not yet an instance
def get_transforms(config):
    return {
        "PlantDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "IperDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "Human36mDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
        "TaichiDataset": tt.Compose(
            [
                tt.ToTensor(),
                tt.Lambda(lambda x: (x * 2.0) - 1.0),
            ]
        ),
    }


def get_dataset(config, custom_transforms=None):
    dataset = __datasets__[config["dataset"]]
    if custom_transforms is not None:
        print("Returning dataset with custom transform")
        transforms = custom_transforms
    else:
        transforms = get_transforms(config)[config["dataset"]]
    return dataset, transforms

