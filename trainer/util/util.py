from omegaconf import DictConfig, ListConfig
from trainer.module.STNModule_MNIST import STNModule_MNIST
from trainer.module.STNModuleWmask import STNModuleWmask
from trainer.module.STNModuleWmaskWaffine import STNModuleWmaskWaffine
from trainer.module.STNModuleWmaskWdec import STNModuleWmaskWdec
from trainer.module.STNModuleWmaskWpix import STNModuleWmaskWpix
from trainer.data.MNISTDatasetModule import MNISTDatasetModule
from trainer.data.BlenderDatasetModule import BlenderDatasetModule
from trainer.data.BlenderDatasetModule_E import BlenderDatasetModule_E
# from trainer.data.BlenderDatasetModule_E2 import BlenderDatasetModule_E2

MODEL_REGISTRY = {
    "SpatialTransformerNetwork": STNModule_MNIST,
    "SpatialTransformerNetworkWmask": STNModuleWmask,
    "SpatialTransformerNetworkWmaskWdec": STNModuleWmaskWdec,
    "SpatialTransformerNetworkWmaskWpix": STNModuleWmaskWpix,
    "SpatialTransformerNetworkWmaskWaffine": STNModuleWmaskWaffine,
}   
DATA_REGISTRY = {
    "MNIST": MNISTDatasetModule,
    "BlenderDataset": BlenderDatasetModule,
    "BlenderDataset_E": BlenderDatasetModule_E,
#     "BlenderDataset_E2": BlenderDatasetModule_E2,
}

def get_model(model_cfg: DictConfig, optim_cfg: DictConfig):
    assert model_cfg.type in MODEL_REGISTRY, f"Model type {model_cfg.type} not found in MODEL_REGISTRY."
    model_class = MODEL_REGISTRY[model_cfg.type]
    model = model_class(model_cfg, optim_cfg)
    return model

def get_data(data_cfg: ListConfig):
    assert data_cfg is not None, "Data configuration must be provided."
    assert isinstance(data_cfg, (list, ListConfig)), "Data configuration must be a list of configurations."
    
    if len(data_cfg) != 1:
        raise NotImplementedError("Currently, only a single data configuration is supported.")
    else:
        data_cfg = data_cfg[0]
        assert data_cfg.type in DATA_REGISTRY, f"Data type {data_cfg.type} not found in DATA_REGISTRY."
        data_class = DATA_REGISTRY[data_cfg.type]
        data = data_class(**data_cfg.params)
    return data