# --------------------------------------------------------
# InternVL
# Copyright (c) 2025 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from packaging import version

import transformers
import torch
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.functional import liger_cross_entropy
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.swiglu import LigerBlockSparseTop2MLP
from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss, )

from liger_kernel.transformers.monkey_patch import (_bind_method_to_module,
                                                    _patch_rms_norm_module)

from internvl.model.internlm2.modeling_internlm2 import _CONFIG_FOR_DOC, InternLM2_INPUTS_DOCSTRING

transformer_version = version.parse(transformers.__version__)

logger = logging.getLogger(__name__)


@add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast,
                           config_class=_CONFIG_FOR_DOC)
def internlm2_lce_forward_deprecated(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, InternLM2ForCausalLM

    >>> model = InternLM2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = (output_attentions if output_attentions is not None
                         else self.config.output_attentions)
    output_hidden_states = (output_hidden_states if output_hidden_states
                            is not None else self.config.output_hidden_states)
    return_dict = (return_dict
                   if return_dict is not None else self.config.use_return_dict)

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         position_ids=position_ids,
                         past_key_values=past_key_values,
                         inputs_embeds=inputs_embeds,
                         use_cache=use_cache,
                         output_attentions=output_attentions,
                         output_hidden_states=output_hidden_states,
                         return_dict=return_dict)

    hidden_states = outputs[0]

    loss = None
    logits = None

    if self.training and (labels is not None):
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # flatten tokens
        shift_hidden_states = shift_hidden_states.view(-1,
                                                       self.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        lce = LigerFusedLinearCrossEntropyLoss()
        loss = lce(self.output.weight, shift_hidden_states, shift_labels)
    else:
        logits = self.output(hidden_states)
        logits = logits.float()
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits, ) + outputs[1:]
        return (loss, ) + output if loss is not None else output

    device = input_ids.device if input_ids is not None else inputs_embeds.device
    output = CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    output['logits'] = output['logits'].to(device)
    return output


def apply_liger_kernel_to_internlm2(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Llama models (2 and 3)

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from internvl.model.internlm2 import modeling_internlm2
    from internvl.model.internlm2.modeling_internlm2 import InternLM2Model

    if rope:
        modeling_internlm2.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_internlm2.InternLM2RMSNorm = LigerRMSNorm
    if swiglu:
        modeling_internlm2.InternLM2MLP = LigerBlockSparseTop2MLP

    if cross_entropy:
        modeling_internlm2.CrossEntropyLoss = LigerCrossEntropyLoss

    if fused_linear_cross_entropy:
        modeling_internlm2.InternLM2ForCausalLM.forward = internlm2_lce_forward_deprecated

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules (e.g. LlamaRMSNorm or LlamaMLP)

        # get the base model from the model instance
        base_model: InternLM2Model = getattr(model, model.base_model_prefix,
                                             model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _bind_method_to_module(decoder_layer.feed_forward, "forward",
                                       LigerBlockSparseTop2MLP.forward)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.attention_norm)
                _patch_rms_norm_module(decoder_layer.ffn_norm)
