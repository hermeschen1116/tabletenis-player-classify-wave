from typing import Optional, Tuple

import torch


class LSTMClassifier(torch.nn.Module):
	def __init__(
		self,
		lstm_hidden_size: int,
		lstm_num_layers: int = 1,
		lstm_bias: bool = True,
		lstm_dropout: float = 0.0,
		lstm_proj_size: int = 0,
		output_head_bias: bool = True,
		device: Optional[str] = None,
		dtype: torch.dtype = torch.bfloat16,
	):
		super(LSTMClassifier, self).__init__()

		self.lstm = torch.nn.LSTM(
			input_size=6,
			hidden_size=lstm_hidden_size,
			num_layers=lstm_num_layers,
			bias=lstm_bias,
			batch_first=True,
			dropout=lstm_dropout,
			bidirectional=False,
			proj_size=lstm_proj_size,
			device=device,
			dtype=dtype,
		)

		self.gender_head = torch.nn.Linear(lstm_hidden_size, 2, bias=output_head_bias, device=device, dtype=dtype)
		self.hold_racket_handed_head = torch.nn.Linear(
			lstm_hidden_size, 2, bias=output_head_bias, device=device, dtype=dtype
		)
		self.play_years_head = torch.nn.Linear(lstm_hidden_size, 3, bias=output_head_bias, device=device, dtype=dtype)
		self.level_head = torch.nn.Linear(lstm_hidden_size, 3, bias=output_head_bias, device=device, dtype=dtype)

	def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		gender_head_output, hold_racket_handed_head_output, play_years_head_output, level_head_output = self.forward(x)
		return (
			gender_head_output.softmax(dim=1).argmax(dim=1),
			hold_racket_handed_head_output.softmax(dim=1).argmax(dim=1),
			play_years_head_output.softmax(dim=1).argmax(dim=1),
			level_head_output.softmax(dim=1).argmax(dim=1),
		)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		lstm_output, _ = self.lstm(x)
		lstm_output = lstm_output[:, -1, :]

		gender_head_output = self.gender_head(lstm_output)
		hold_racket_handed_head_output = self.hold_racket_handed_head(lstm_output)
		play_years_head_output = self.play_years_head(lstm_output)
		level_head_output = self.level_head(lstm_output)

		return gender_head_output, hold_racket_handed_head_output, play_years_head_output, level_head_output
