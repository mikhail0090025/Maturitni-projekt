import torch
from torch import nn, optim
import json
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import AdamW

# -----------------------------
# Блоки
# -----------------------------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1, groups=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, groups=groups)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3, padding=1, norm="batch", activation="relu"):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=scale_factor, padding=padding, output_padding=scale_factor - 1
        )
        self.norm = nn.BatchNorm2d(out_channels) if norm=="batch" else \
                    nn.InstanceNorm2d(out_channels) if norm=="instance" else None
        self.act = nn.ReLU(inplace=True) if activation=="relu" else \
                   nn.LeakyReLU(0.01, inplace=True) if activation=="leaky_relu" else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.act: x = self.act(x)
        return x

class ConvolutionalAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, in_channels)
        self.attention = nn.MultiheadAttention(in_channels, num_heads, batch_first=True, dropout=0.1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h*w).permute(0,2,1)
        attn, _ = self.attention(x_flat, x_flat, x_flat)
        x_flat = x_flat + attn * 0.1
        x_out = x_flat.permute(0,2,1).contiguous().view(b,c,h,w)
        x_out = self.norm(x_out)
        return x_out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, depth=2, norm="batch", activation="relu"):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Conv2d(
                in_channels if i==0 else out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))
            if norm=="batch": layers.append(nn.BatchNorm2d(out_channels))
            elif norm=="instance": layers.append(nn.InstanceNorm2d(out_channels))
            if activation=="relu": layers.append(nn.ReLU(inplace=True))
            elif activation=="leaky_relu": layers.append(nn.LeakyReLU(0.01, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)

# -----------------------------
# Функция создания модели
# -----------------------------

def create_model(json_text):
    layers_list = []
    architecture = json.loads(json_text)
    last_out_size = 1

    for i, layer in enumerate(architecture):
        layer_type = layer['type']
        params = layer.get('params', {})

        # --- Полносвязные ---
        if layer_type=="Linear":
            layers_list.append(nn.Linear(params["in_features"], params["out_features"]))
            last_out_size = params["out_features"]

        # --- Свёрточные ---
        elif layer_type=="Conv2D":
            layers_list.append(nn.Conv2d(
                in_channels=params["in_features"],
                out_channels=params["out_features"],
                kernel_size=params["kernel_size"],
                stride=params.get("stride",1),
                padding=params.get("padding",0)
            ))
            last_out_size = params["out_features"]

        elif layer_type=="DSConv2D":
            layers_list.append(DepthwiseSeparableConv(
                params["in_features"],
                params["out_features"],
                kernel_size=params.get("kernel_size",3),
                stride=params.get("stride",1),
                padding=params.get("padding",1)
            ))
            last_out_size = params["out_features"]

        # --- Upscale ---
        elif layer_type=="Conv2DTranspose":
            layers_list.append(nn.ConvTranspose2d(
                in_channels=params["in_features"],
                out_channels=params["out_features"],
                kernel_size=params.get("kernel_size",2),
                stride=params.get("stride",1),
                padding=params.get("padding",0),
                output_padding=params.get("stride",1)-1
            ))
            last_out_size = params["out_features"]

        elif layer_type=="UpscaleBlock":
            layers_list.append(UpscaleBlock(
                in_channels=params["in_features"],
                out_channels=params["out_features"],
                scale_factor=params.get("scale_factor",2),
                kernel_size=params.get("kernel_size",3),
                padding=params.get("padding",1),
                norm=params.get("norm","batch"),
                activation=params.get("activation","relu")
            ))
            last_out_size = params["out_features"]

        elif layer_type=="Upsample":
            layers_list.append(nn.Upsample(
                scale_factor=params.get("scale_factor",2),
                mode=params.get("mode","nearest")
            ))

        # --- Attention ---
        elif layer_type=="ConvolutionalAttention":
            layers_list.append(ConvolutionalAttention(
                in_channels=params["in_features"],
                num_heads=params.get("num_heads",8)
            ))
            last_out_size = params["in_features"]

        # --- Residual ---
        elif layer_type=="ResidualBlock":
            layers_list.append(ResidualBlock(
                in_channels=params["in_features"],
                out_channels=params["out_features"],
                kernel_size=params.get("kernel_size",3),
                stride=params.get("stride",1),
                padding=params.get("padding",1),
                depth=params.get("depth",2),
                norm=params.get("norm","batch"),
                activation=params.get("activation","relu")
            ))
            last_out_size = params["out_features"]

        # --- Пулинг ---
        elif layer_type=="MaxPool2D":
            layers_list.append(nn.MaxPool2d(
                kernel_size=params["kernel_size"],
                stride=params.get("stride",None),
                padding=params.get("padding",0)
            ))

        elif layer_type=="AvgPool2D":
            layers_list.append(nn.AvgPool2d(
                kernel_size=params["kernel_size"],
                stride=params.get("stride",None),
                padding=params.get("padding",0)
            ))

        # --- Нормализация ---
        elif layer_type=="BatchNorm1d": layers_list.append(nn.BatchNorm1d(last_out_size))
        elif layer_type=="BatchNorm2d": layers_list.append(nn.BatchNorm2d(last_out_size))
        elif layer_type=="LayerNorm": layers_list.append(nn.LayerNorm(last_out_size))

        # --- Активации ---
        elif layer_type=="ReLU": layers_list.append(nn.ReLU())
        elif layer_type=="LeakyReLU": layers_list.append(nn.LeakyReLU(params.get("alpha",0.01)))
        elif layer_type=="Sigmoid": layers_list.append(nn.Sigmoid())
        elif layer_type=="Tanh": layers_list.append(nn.Tanh())
        elif layer_type=="Softmax": layers_list.append(nn.Softmax(dim=params.get("dim",1)))

        # --- Регуляризация ---
        elif layer_type=="Dropout": layers_list.append(nn.Dropout(params.get("p",0.5)))

        # --- Flatten / Reshape ---
        elif layer_type=="Flatten": layers_list.append(nn.Flatten())
        elif layer_type=="Reshape":
            layers_list.append(nn.Unflatten(1, tuple(params["shape"])))

        else:
            raise ValueError(f"Неизвестный тип слоя: {layer_type}")

    model = nn.Sequential(*layers_list)
    print("Created model:", model)
    return model

def create_optimizer(model, optimizer_json):
    """
    optimizer_json example:
    {
        "type": "AdamW",
        "lr": 0.0003,
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999
    }
    """
    optimizer_json = json.loads(optimizer_json)
    print(optimizer_json, type(optimizer_json))
    print("-------------------------------")
    opt_type = optimizer_json.get("type", "AdamW")

    if opt_type != "AdamW":
        raise ValueError(f"Unsupported optimizer: {opt_type}")

    return AdamW(
        model.parameters(),
        lr=optimizer_json.get("lr", 3e-4),
        weight_decay=optimizer_json.get("weight_decay", 0.0),
        betas=(
            optimizer_json.get("beta1", 0.9),
            optimizer_json.get("beta2", 0.999),
        )
    )

def create_scheduler(optimizer, scheduler_json):
    """
    scheduler_json examples:

    Cosine:
    {
        "type": "cosine",
        "total_steps": 20000,
        "min_lr": 1e-6,
        "warmup_steps": 500
    }

    Plateau:
    {
        "type": "plateau",
        "mode": "min",
        "factor": 0.1,
        "patience": 5,
        "min_lr": 1e-6
    }
    """
    scheduler_json = json.loads(scheduler_json)
    sched_type = scheduler_json.get("type", "none")

    if sched_type == "none":
        return None

    if sched_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_json.get("total_steps", 1000),
            eta_min=scheduler_json.get("min_lr", 1e-6)
        )

    if sched_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=scheduler_json.get("mode", "min"),
            factor=scheduler_json.get("factor", 0.1),
            patience=scheduler_json.get("patience", 5),
            min_lr=scheduler_json.get("min_lr", 1e-6)
        )

    raise ValueError(f"Unsupported scheduler: {sched_type}")

# -----------------------------
# Класс для обучения
# -----------------------------
class FullModel:
    def __init__(self, model, optimizer, scheduler, criterion=nn.MSELoss()):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
        }, path)
    
    def forward(self, x):
        return self.model(x)
    
    def go_epochs(self, data_loader, epochs=1.0, device='cpu'):
        self.model.to(device)
        self.model.train()
        steps = int(epochs * len(data_loader))
        for step in range(steps):
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"Step [{step+1}/{steps}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}")

            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()

    def go_steps(self, data_loader, steps=1000, device='cpu'):
        self.model.to(device)
        self.model.train()
        step_count = 0
        while step_count < steps:
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if step_count >= steps:
                    break
                inputs, targets = inputs.to(device), targets.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"Step [{step_count+1}/{steps}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}")
                step_count += 1

            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()

    def evaluate(self, data_loader, device='cpu'):
        self.model.to(device)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss
    
    def predict(self, x, device='cpu'):
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            x = x.to(device)
            outputs = self.model(x)
        return outputs
    
    def to(self, device):
        self.model.to(device)
        self.criterion.to(device)

    def load(self, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'criterion_state_dict' in checkpoint:
            self.criterion.load_state_dict(checkpoint['criterion_state_dict'])

def create_full_model(
    architecture_json: str,
    optimizer_json: str,
    scheduler_json: str,
    criterion_name: str = "MSELoss"
):
    # --- model ---
    model = create_model(architecture_json)
    print("Model created.")

    # --- optimizer ---
    optimizer_cfg = json.loads(optimizer_json)
    optimizer = create_optimizer(model, optimizer_cfg)

    # --- scheduler ---
    scheduler_cfg = json.loads(scheduler_json)
    scheduler = create_scheduler(optimizer, scheduler_cfg)

    # --- loss ---
    criterion = getattr(nn, criterion_name)()

    return FullModel(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion
    )