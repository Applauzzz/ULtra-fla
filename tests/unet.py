import torch

# from fla.models.U_Net.modeling_Utransformer import TransformerForCausalLM
# from fla.models.U_Net.configuration_Utransformer import TransformerConfig

from fla.models.transformer_sb import TransformerConfig
from fla.models.transformer_sb import TransformerForCausalLM
import pdb

if __name__ == "__main__":
    config = TransformerConfig()
    print("start to model")
    model = TransformerForCausalLM(config).to(torch.bfloat16).cuda()
    print("start to input")
    input_ids = torch.randint(0, 32000, (1, 2045), dtype=torch.long).cuda()
    do = torch.randn((1, 2045, 32000), device=input_ids.device, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    loss = model(input_ids=input_ids, labels=input_ids).loss
    print(input_ids.shape)
    print("start to generate")
    # loss = (output * do).sum()
    loss.backward()
    
    # output = model.generate(inputs=input_ids, max_new_tokens=15, )
    # out_para = model(input_ids=output)
    pdb.set_trace()