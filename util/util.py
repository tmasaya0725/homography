from omegaconf import DictConfig
from trainer.model.STNModule import STNModule
from trainer.data.MNISTDataModule import MNISTDataModule

MODEL_REGISTRY = {
    "SpatialTransformerNetwork": STNModule,
}
DATA_REGISTRY = {
    "MNIST": MNISTDataModule,
}

def get_model(model_cfg: DictConfig, optim_cfg: DictConfig):
    assert model_cfg.type in MODEL_REGISTRY, f"Model type {model_cfg.type} not found in MODEL_REGISTRY."
    model_class = MODEL_REGISTRY[model_cfg.type]
    model = model_class(model_cfg, optim_cfg)
    return model

def get_data(data_cfg: DictConfig):
    assert data_cfg.type in DATA_REGISTRY, f"Data type {data_cfg.type} not found in DATA_REGISTRY."
    data_class = DATA_REGISTRY[data_cfg.type]
    data = data_class(**data_cfg.params)
    return data

