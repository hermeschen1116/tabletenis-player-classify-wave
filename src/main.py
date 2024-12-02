import multiprocessing
from typing import List

import polars
import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tabletenis_player_classify_wave.Model import LSTMClassifier

num_workers: int = multiprocessing.cpu_count()
device: str = "cuda"
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
	index_column="time_order", group_by=["player_ID", "data_id"], period=f"{series_length - 1}i", closed="both"
).agg(polars.all())

train_data_df = (
	train_data_df.with_columns(polars.col(features).list.get(0).name.suffix("_PAD"))
	.with_columns(
		polars.concat_list(polars.col(f"{col}_PAD").repeat_by(series_length - polars.col(col).list.len()), col).alias(
			col
		)
		for col in features
	)
	.drop("^[AG][xyz]_PAD$")
)

train_df = train_data_df.join(train_info_df, on="data_id").drop(["player_ID", "data_id", "time_order"])

dataset = Dataset(train_df.to_arrow())

dataset = dataset.map(
	lambda sample: {"data": torch.stack(tuple(torch.tensor(sample[col]) for col in features), dim=1)},
	remove_columns=features,
)

dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.with_format("torch")

train_dataloader = DataLoader(
	dataset["train"],
	batch_size=4,
	num_workers=num_workers,
	pin_memory=True,
	multiprocessing_context="spawn",
	persistent_workers=True,
)
validation_dataloader = DataLoader(
	dataset["test"],
	batch_size=4,
	num_workers=num_workers,
	pin_memory=True,
	multiprocessing_context="spawn",
	persistent_workers=True,
)

num_epochs: int = 5
learning_rate: float = 0.0001

model = LSTMClassifier(42, device="cuda")

loss_fn = CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(num_epochs):
	model.train()
	training_loss: torch.Tensor = torch.empty(0)
	for batch in tqdm(train_dataloader, desc=f"Train(Epoch {i}): ", colour="green"):
		data = batch["data"].bfloat16().to(device)
		label = tuple(batch[col].long().to(device) for col in labels)

		optimizer.zero_grad()
		outputs = model.forward(data)
		loss = torch.tensor([loss_fn(outputs[j], label[j]) for j in range(4)], requires_grad=True).sum().unsqueeze(0)
		training_loss = torch.cat((training_loss, loss))
		loss.backward()
		optimizer.step()
	print(f"Training Loss(Epoch {i}): {training_loss.mean().item()}")
	validation_loss: torch.Tensor = torch.empty(0)
	predictions = [torch.empty(0) for _ in range(4)]
	truths = [torch.empty(0) for _ in range(4)]
	with torch.no_grad():
		model.eval()
		count = 0
		for batch in tqdm(validation_dataloader, desc=f"Validation(Epoch {i}): ", colour="blue"):
			data = batch["data"].bfloat16()
			label = tuple(batch[col].long() for col in labels)

			outputs = model.forward(data.to(device))
			loss = (
				torch.tensor([loss_fn(outputs[j], label[j].to(device)) for j in range(4)], requires_grad=True)
				.sum()
				.unsqueeze(0)
			)
			validation_loss = torch.cat((validation_loss, loss))

			truths = [torch.cat((truths[i], label[i])) for i in range(4)]
			predictions = [torch.cat((predictions[i], torch.max(outputs[i].cpu(), dim=-1)[0])) for i in range(4)]
			count += 1
	print(f"Validation Loss(Epoch {i}): {validation_loss.mean().item()}")
	gender_score = roc_auc_score(truths[0].numpy(), predictions[0].numpy(), average="micro")
	hold_racket_handed_score = roc_auc_score(truths[1].numpy(), predictions[1].numpy(), average="micro")
	play_years_score = roc_auc_score(truths[2].numpy(), predictions[2].numpy(), average="micro", multi_class="ovr")
	level_score = roc_auc_score(truths[3].numpy(), predictions[3].numpy(), average="micro", multi_class="ovr")
	avg_score = (gender_score + hold_racket_handed_score + play_years_score + level_score) / 4
	print(f"""
        ROC AUC Score
        Gender: {gender_score}
        Hold racket handed: {hold_racket_handed_score}
        Player years: {play_years_score}
        Level: {level_score}
        Average Score: {avg_score}
        """)

test_df = polars.read_csv(
	"../data/test_data.csv", n_threads=num_workers, low_memory=True, rechunk=True, use_pyarrow=True
)
test_df = test_df.sort("data_id", "time_order").set_sorted("time_order")

test_df = test_df.with_columns((polars.col(features) - polars.col(features).mean()) / polars.col(features).std())

test_df = test_df.rolling(
	index_column="time_order", group_by="data_id", period=f"{series_length - 1}i", closed="both"
).agg(polars.all())

test_df = (
	test_df.with_columns(polars.col(features).list.get(0).name.suffix("_PAD"))
	.with_columns(
		polars.concat_list(polars.col(f"{col}_PAD").repeat_by(series_length - polars.col(col).list.len()), col).alias(
			col
		)
		for col in features
	)
	.drop("^[AG][xyz]_PAD$")
)

test_df = test_df.drop("time_order")

test_dataset = Dataset(test_df.to_arrow())

test_dataset = test_dataset.map(
	lambda sample: {"data": torch.stack(tuple(torch.tensor(sample[col]) for col in features), dim=1)},
	remove_columns=features,
)

test_dataset = test_dataset.with_format("torch", columns=["data"], output_all_columns=True)

test_dataset = test_dataset.map(
	lambda sample: {"output": model(sample.unsqueeze(0).to(device))}, input_columns=["data"], remove_columns=["data"]
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
