# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.vnsa.configuration_nsa import VNSAConfig
from fla.models.vnsa.modeling_nsa import VNSAForCausalLM, VNSAModel

# AutoConfig.register(NSAConfig.model_type, NSAConfig, exist_ok=True)
# AutoModel.register(NSAConfig, NSAModel, exist_ok=True)
# AutoModelForCausalLM.register(NSAConfig, NSAForCausalLM, exist_ok=True)


__all__ = [
    'VNSAConfig', 'VNSAModel', 'VNSAForCausalLM',
]
