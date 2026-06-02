"""A minimal input regression baseline model."""

import torch
import torch.nn as nn

from deepks.io.checkpoints import load_compiled_or_checkpoint, save_model_checkpoint
from deepks.ml.base import ModelAdapter

SCALE_EPS = 1e-8


class LinearModel(nn.Module, ModelAdapter):

    MODEL_FAMILY = "linear"

    def __init__(
        self,
        input_dim,
        input_shift=0,
        input_scale=1,
        output_scale=1,
    ):
        super().__init__()
        self._init_args = {
            "input_dim": input_dim,
            "input_shift": input_shift,
            "input_scale": input_scale,
            "output_scale": output_scale,
        }
        self.input_dim = input_dim
        self.input_partition = None
        self.embedder = None
        self.linear = nn.Linear(input_dim, 1).double()
        self.input_shift = nn.Parameter(
            torch.tensor(input_shift, dtype=torch.float64).expand(input_dim).clone(),
            requires_grad=False,
        )
        self.input_scale = nn.Parameter(
            torch.tensor(input_scale, dtype=torch.float64).expand(input_dim).clone(),
            requires_grad=False,
        )
        self.output_scale = nn.Parameter(
            torch.tensor(output_scale, dtype=torch.float64),
            requires_grad=False,
        )
        self.output_bias = nn.Parameter(
            torch.tensor(0, dtype=torch.float64),
            requires_grad=False,
        )

    def forward(self, model_inputs):
        """Return the unreduced per-atom contribution (see CorrNet.forward)."""

        from deepks.ml.models.corrnet import _extract_descriptor_tensor

        x = _extract_descriptor_tensor(model_inputs)
        x = (x - self.input_shift) / (self.input_scale + SCALE_EPS)
        return self.linear(x) / self.output_scale

    def set_normalization(self, shift=None, scale=None):
        dtype = self.input_scale.dtype
        if shift is not None:
            self.input_shift.data[:] = torch.tensor(shift, dtype=dtype)
        if scale is not None:
            self.input_scale.data[:] = torch.tensor(scale, dtype=dtype)

    def set_prefitting(self, weight, bias, trainable=False):
        dtype = self.linear.weight.dtype
        self.linear.weight.data[:] = torch.tensor(weight, dtype=dtype).reshape(-1)
        self.linear.bias.data[:] = torch.tensor(bias, dtype=dtype).reshape(-1)
        self.linear.requires_grad_(trainable)

    def set_output_bias(self, const):
        dtype = self.output_bias.dtype
        self.output_bias.data = torch.tensor(const, dtype=dtype).reshape([])

    def save_dict(self, **extra_info):
        return {
            "model_family": self.MODEL_FAMILY,
            "state_dict": self.state_dict(),
            "init_args": self._init_args,
            "extra_info": extra_info,
        }

    def save(self, filename, **extra_info):
        save_model_checkpoint(filename, self.save_dict(**extra_info))

    def compile(self, set_eval=True, **kwargs):
        old_mode = self.training
        if set_eval:
            self.eval()
        # See CorrNet.compile docstring: tracing a nn.Module wrapper rather
        # than a closure prevents the "requires_grad as constant" trace error.
        from deepks.ml.models.corrnet import _DescriptorEnergyTraceWrapper

        wrapper = _DescriptorEnergyTraceWrapper(self).eval()
        smodel = torch.jit.trace(
            wrapper,
            torch.empty((2, 2, self.input_dim)),
            **kwargs,
        )
        self.train(old_mode)
        return smodel

    def compile_save(self, filename, **kwargs):
        torch.jit.save(self.compile(**kwargs), filename)

    @staticmethod
    def load_dict(checkpoint, strict=False):
        init_args = dict(checkpoint["init_args"])
        model = LinearModel(**init_args)
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        return model

    @staticmethod
    def load(filename, strict=False):
        checkpoint = load_compiled_or_checkpoint(filename)
        if not isinstance(checkpoint, dict):
            return checkpoint
        return LinearModel.load_dict(checkpoint, strict=strict)
