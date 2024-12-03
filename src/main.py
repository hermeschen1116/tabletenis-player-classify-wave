import multiprocessing
import os
import random
from typing import List

import numpy
import polars
import torch
from datasets import Dataset
from dotenv import load_dotenv
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tabletenis_player_classify_wave.Model import LSTMClassifier

load_dotenv()

seed: int = 42
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = multiprocessing.cpu_count()
device: str = "cuda"
dtype: torch.dtype = torch.float32
features: List[str] = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
labels: List[str] = ["gender", "hold racket handed", "play years", "level"]

train_info_df = polars.read_csv(
	f"{project_root}/data/train_info.csv", n_threads=num_workers, low_memory=True, rechunk=True, use_pyarrow=True
)
train_data_df = polars.read_csv(
	f"{project_root}/data/train_data.csv", n_threads=num_workers, low_memory=True, rechunk=True, use_pyarrow=True
)

train_data_df = train_data_df.group_by(["data_id", "player_ID"]).agg(polars.all())

train_df = train_data_df.join(train_info_df, on="data_id").drop(["player_ID", "data_id", "time_order"])

dataset = Dataset(train_df.to_arrow())

dataset = dataset.map(
	lambda sample: {"data": torch.stack(tuple(torch.tensor(sample[col]) for col in features), dim=1)},
	remove_columns=features,
)

dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.with_format("torch")

train_dataloader = DataLoader(dataset["train"])
validation_dataloader = DataLoader(dataset["test"])

num_epochs = [20, 3, 25, 25]
learning_rate: float = 1e-3

models = [
	LSTMClassifier(input_size=6, lstm_hidden_size=16, lstm_num_layers=3, output_features=2, device="cuda"),
	LSTMClassifier(input_size=6, lstm_hidden_size=32, lstm_num_layers=2, output_features=2, device="cuda"),
	LSTMClassifier(input_size=6, lstm_hidden_size=16, lstm_num_layers=3, output_features=3, device="cuda"),
	LSTMClassifier(input_size=6, lstm_hidden_size=16, lstm_num_layers=3, output_features=3, device="cuda"),
]

loss_fn = CrossEntropyLoss()
optimizers = [torch.optim.AdamW(models[i].parameters(), lr=learning_rate, weight_decay=1e-4) for i in range(4)]
learning_rate_scheduler = [ConstantLR(optimizers[i]) for i in range(4)]

for m in range(4):
	for i in range(num_epochs[m]):
		models[m].train()
		training_loss: torch.Tensor = torch.empty(0)
		print(f"Learning Rate({labels[m]} classifier, Epoch {i}): {learning_rate_scheduler[m].get_last_lr()[0]}")
		for batch in tqdm(train_dataloader, desc=f"Train({labels[m]} classifier, Epoch {i}): ", colour="green"):
			data = batch["data"].to(dtype=dtype, device=device)
			label = batch[labels[m]].to(dtype=torch.long, device=device)

			optimizers[m].zero_grad()
			outputs = models[m].forward(data)
			loss = loss_fn(outputs, label)
			training_loss = torch.cat((training_loss, loss.unsqueeze(0).cpu()))
			loss.backward()
			optimizers[m].step()
		print(f"Training Loss(Epoch {i}): {training_loss.mean().item()}")
		validation_loss: torch.Tensor = torch.empty(0)
		with torch.no_grad():
			models[m].eval()
			for batch in tqdm(
				validation_dataloader, desc=f"Validation({labels[m]} classifier, Epoch {i}): ", colour="blue"
			):
				data = batch["data"].to(dtype=dtype, device=device)
				label = batch[labels[m]].to(dtype=torch.long, device=device)

				outputs = models[m].forward(data.to(device))
				loss = loss_fn(outputs, label)
				learning_rate_scheduler[m].step()
				validation_loss = torch.cat((validation_loss, loss.unsqueeze(0).cpu()))

		print(f"Validation Loss(Epoch {i}): {validation_loss.mean().item()}")

test_df = polars.read_csv(
	f"{project_root}/data/test_data.csv", n_threads=num_workers, low_memory=True, rechunk=True, use_pyarrow=True
)
test_df = test_df.sort("data_id", "time_order").set_sorted("time_order")

test_df = test_df.group_by("data_id").agg(polars.all())

test_df = test_df.drop("time_order")

test_dataset = Dataset(test_df.to_arrow())

test_dataset = test_dataset.map(
	lambda sample: {"data": torch.stack(tuple(torch.tensor(sample[col]) for col in features), dim=1) for i in range(4)},
	remove_columns=features,
)
test_dataset = test_dataset.with_format("torch", columns=["data"], output_all_columns=True)

test_dataset = test_dataset.map(
	lambda sample: {
		"output": [models[i](sample["data"].unsqueeze(0).to(dtype=dtype, device=device)) for i in range(4)]
	},
	remove_columns=[f"data{i}" for i in range(4)],
)

test_dataset = test_dataset.map(
	lambda sample: {
		"gender": sample[0].item(),
		"hold racket handed": sample[1].item(),
		"play years_0": int(sample[2].item() == 0),
		"play years_1": int(sample[2].item() == 1),
		"play years_2": int(sample[2].item() == 2),
		"level_0": int(sample[3].item() == 0),
		"level_1": int(sample[3].item() == 1),
		"level_2": int(sample[3].item() == 2),
	},
	input_columns=["output"],
	remove_columns=["output"],
)

test_dataset.to_csv(f"{project_root}/data/result.csv")
