import os
from typing import List

import polars
import torch
from datasets import Dataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from tabletenis_player_classify_wave.Model import LSTMClassifier

num_workers: int = os.cpu_count()
features: List[str] = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
labels: List[str] = ["gender", "hold racket handed", "play years", "level"]

train_info_df = polars.read_csv(
	"../data/train_info.csv", n_threads=num_workers, low_memory=True, rechunk=True, use_pyarrow=True
)
train_data_df = polars.read_csv(
	"../data/train_data.csv", n_threads=num_workers, low_memory=True, rechunk=True, use_pyarrow=True
)

train_data_df = (
	train_data_df.sort("player_ID", "data_id", "time_order")
	.set_sorted("time_order")
	.group_by("player_ID", maintain_order=True)
	.agg([polars.col(features).mean().name.suffix("_mean"), polars.col(features).std().name.suffix("_std")])
	.join(train_data_df, on="player_ID")
	.with_columns(((polars.col(col) - polars.col(f"{col}_mean")) / polars.col(f"{col}_std")) for col in features)
	.drop(["^[AG][xyz]_mean$", "^[AG][xyz]_std$"])
)

series_length: int = 42

train_data_df = train_data_df.rolling(
	index_column="time_order",
	group_by=["player_ID", "data_id"],
	offset="0i",
	period=f"{series_length - 1}i",
	closed="both",
).agg(polars.all())

train_df = train_data_df.join(train_info_df, on="data_id").drop(["player_ID", "data_id", "time_order"])

dataset = Dataset(train_df.to_arrow())

dataset = dataset.map(
	lambda sample: {"data": torch.stack(tuple(torch.tensor(sample[col]) for col in features), dim=1)},
	remove_columns=features,
)

dataset = dataset.with_format("torch", format_kwargs={"dtype": torch.bfloat16, "requires_grad": True})

dataset = dataset.train_test_split(test_size=0.2)

batch_size: int = 4

train_dataloader = DataLoader(
	dataset["train"], batch_size=batch_size, pin_memory=True, num_workers=num_workers, persistent_workers=True
)
validation_dataloader = DataLoader(
	dataset["test"], batch_size=batch_size, pin_memory=True, num_workers=num_workers, persistent_workers=True
)

num_epochs: int = 5
learning_rate: float = 0.0001

model = LSTMClassifier(42, device="cuda")
loss_fn = CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(num_epochs):
	model.train()
	for batch in train_dataloader:
		data, label = batch["data"], tuple(batch[col].to("cuda").long() for col in labels)

		optimizer.zero_grad()
		outputs = model(data)
		loss = torch.tensor([loss_fn(outputs[j], label[j]) for j in range(4)]).sum()
		loss.backward()
		optimizer.step()


test_df = polars.read_csv(
	"../data/test_data.csv", n_threads=num_workers, low_memory=True, rechunk=True, use_pyarrow=True
)
test_df = test_df.sort("data_id", "time_order").set_sorted("time_order")

test_df = test_df.with_columns((polars.col(features) - polars.col(features).mean()) / polars.col(features).std())

test_df = test_df.rolling(
	index_column="time_order", group_by="data_id", offset="0i", period=f"{series_length - 1}i", closed="both"
).agg(polars.all())

test_df = test_df.drop("time_order")

test_dataset = Dataset(test_df.to_arrow())

test_dataset = test_dataset.map(
	lambda sample: {"data": torch.stack(tuple(sample[col] for col in features), dim=1)}, remove_columns=features
)

test_dataset.with_format("torch", columns=["data"], output_all_columns=True, format_kwargs={"dtype": torch.bfloat16})

test_dataset = test_dataset.map(
	lambda sample: {"output": model(sample.unsqueeze(0))}, input_columns=["data"], remove_columns=["data"]
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

test_dataset.to_csv("../data/result.csv", num_proc=num_workers)
