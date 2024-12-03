from typing import Optional

import torch


class LSTMClassifier(torch.nn.Module):
	def __init__(
		self,
		input_size,
		lstm_hidden_size: int = 128,
		lstm_num_layers: int = 2,
		lstm_bias: bool = True,
		lstm_proj_size: int = 0,
		dropout: float = 0.3,
		output_features: int = 2,
		output_head_bias: bool = True,
		device: Optional[str] = None,
		dtype: torch.dtype = torch.float32,
	):
		super(LSTMClassifier, self).__init__()

		self.lstm = torch.nn.LSTM(
			input_size=input_size,
			hidden_size=lstm_hidden_size,
			num_layers=lstm_num_layers,
			bias=lstm_bias,
			batch_first=True,
			dropout=0,
			bidirectional=False,
			proj_size=lstm_proj_size,
			device=device,
			dtype=dtype,
		)
		self.dropout = torch.nn.Dropout(p=dropout)

		self.feature_head = torch.nn.Linear(
			lstm_hidden_size, output_features, bias=output_head_bias, device=device, dtype=dtype
		)

	def __call__(self, x: torch.Tensor) -> torch.Tensor:
		return self.forward(x).softmax(dim=1).argmax(dim=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		lstm_output, _ = self.lstm(x)
		lstm_output = lstm_output[:, -1, :]

		head_output = self.feature_head(lstm_output)

		return head_output
