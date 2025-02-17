# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch

from olive.constants import Framework


class RandomDataLoader:
    def __init__(self, create_inputs_func, batch_size, torch_dtype, model_framework=Framework.PYTORCH):
        self.create_input_func = create_inputs_func
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.model_framework = model_framework

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batch_size, self.torch_dtype, self.model_framework), label


def dummy_inputs(batch_size, torch_dtype, model_framework=Framework.PYTORCH, num_hidden_layers=26):
    past_sequence_length = 1
    attention_mask_sequence_length = 1
    sequence_length = 2

    inputs = {
        "input_ids": torch.randint(10, (batch_size, sequence_length), dtype=torch.int64),
        "attention_mask": torch.randint(10, (batch_size, attention_mask_sequence_length), dtype=torch.int64),
    }

    if model_framework == Framework.ONNX:
        for layer_index in range(num_hidden_layers):
            inputs[f"past_key_values.{layer_index}.key"] = torch.rand(
                (batch_size, 32, past_sequence_length, 100), dtype=torch_dtype
            )
            inputs[f"past_key_values.{layer_index}.value"] = torch.rand(
                (batch_size, 32, past_sequence_length, 100), dtype=torch_dtype
            )

        inputs["use_cache_branch"] = torch.ones((1,), dtype=torch.bool)

    return inputs


def dataloader_func(data_dir, batch_size, *args, **kwargs):
    model_framework = kwargs.get("model_framework", Framework.PYTORCH)
    return RandomDataLoader(dummy_inputs, batch_size, torch.float16, model_framework)
