# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.U_Net.configuration_Utransformer import TransformerConfig
from fla.models.U_Net.modeling_Utransformer import TransformerForCausalLM, TransformerModel

# AutoConfig.register(TransformerConfig.model_type, TransformerConfig, exist_ok=True)
# AutoModel.register(TransformerConfig, TransformerModel, exist_ok=True)
# AutoModelForCausalLM.register(TransformerConfig, TransformerForCausalLM, exist_ok=True)


# __all__ = ['TransformerConfig', 'TransformerForCausalLM', 'TransformerModel']
