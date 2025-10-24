import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import json

def create_model(json_text):
    print("JSON text for model creation:")
    print(json_text)
    layers_list = []
    architecture = json.loads(json_text)

    for layer in architecture:
        layer_type = layer['type']
        params = layer.get('params', {})

        # --- Полносвязные слои ---
        if layer_type == 'Linear':
            layers_list.append(
                nn.Linear(params['in_features'], params['out_features'])
            )

        # --- Свёрточные слои ---
        elif layer_type == 'Conv2D':
            layers_list.append(
                nn.Conv2d(
                    in_channels=params['in_channels'],
                    out_channels=params['out_channels'],
                    kernel_size=params['kernel_size'],
                    stride=params.get('stride', 1),
                    padding=params.get('padding', 0)
                )
            )

        # --- Пулинг ---
        elif layer_type == 'MaxPool2D':
            layers_list.append(
                nn.MaxPool2d(
                    kernel_size=params['kernel_size'],
                    stride=params.get('stride', None),
                    padding=params.get('padding', 0)
                )
            )

        elif layer_type == 'AvgPool2D':
            layers_list.append(
                nn.AvgPool2d(
                    kernel_size=params['kernel_size'],
                    stride=params.get('stride', None),
                    padding=params.get('padding', 0)
                )
            )

        # --- Нормализация ---
        elif layer_type == 'BatchNorm1d':
            layers_list.append(nn.BatchNorm1d(params['num_features']))
        elif layer_type == 'BatchNorm2d':
            layers_list.append(nn.BatchNorm2d(params['num_features']))
        elif layer_type == 'LayerNorm':
            layers_list.append(nn.LayerNorm(params['normalized_shape']))

        # --- Активации ---
        elif layer_type == 'ReLU':
            layers_list.append(nn.ReLU())
        elif layer_type == 'LeakyReLU':
            layers_list.append(nn.LeakyReLU(params.get('negative_slope', 0.01)))
        elif layer_type == 'Sigmoid':
            layers_list.append(nn.Sigmoid())
        elif layer_type == 'Tanh':
            layers_list.append(nn.Tanh())
        elif layer_type == 'Softmax':
            layers_list.append(nn.Softmax(dim=params.get('dim', 1)))

        # --- Регуляризация ---
        elif layer_type == 'Dropout':
            layers_list.append(nn.Dropout(params.get('p', 0.5)))

        # --- Служебные ---
        elif layer_type == 'Flatten':
            layers_list.append(nn.Flatten())

        elif layer_type == 'Reshape':
            layers_list.append(
                nn.Unflatten(1, tuple(params['shape']))  # обратный Flatten
            )

        else:
            raise ValueError(f"Неизвестный тип слоя: {layer_type}")

    # Собираем последовательную модель
    model = nn.Sequential(*layers_list)
    return model

class FullModel:
    def __init__(self, model, criterion=nn.MSELoss(), lr=0.001, patience=5, factor=0.1, weight_decay=1e-5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=patience, factor=factor)
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
        }, path)