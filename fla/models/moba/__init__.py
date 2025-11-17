# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.moba.configuration_transformer import MobaTransformerConfig
from fla.models.moba.modeling_transformer import MobaTransformerForCausalLM, MobaTransformerModel

# AutoConfig.register(TransformerConfig.model_type, TransformerConfig, exist_ok=True)
# AutoModel.register(TransformerConfig, TransformerModel, exist_ok=True)
# AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)


__all__ = ['MobaTransformerConfig', 'MobaTransformerForCausalLM', 'MobaTransformerModel']
