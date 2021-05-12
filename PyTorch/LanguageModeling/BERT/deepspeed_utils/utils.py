import deepspeed as dp
import torch

def prepare_optimizer_parameters(args, model):

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args.deepspeed_transformer_kernel:
        no_decay = no_decay + ['attn_nw', 'attn_nb', 'norm_w', 'norm_b',
                               'attn_qkvb', 'attn_ob', 'inter_b', 'output_b']
    weight_decay = 0.01

    optimizer_grouped_parameters = [{
        'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
            weight_decay
    }, {
        'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]

    return optimizer_grouped_parameters


def initialize_deepspeed(args, model):

    print("SystemLog: Initializing DeepSpeed")
    print("SystemLog: DeepSpeed parameters: deepspeed_config=%s" % (args.deepspeed_config))

    optimizer_grouped_parameters = prepare_optimizer_parameters(args, model)

    # DeepSpeed initializer handles FP16, distributed, optimizer automatically.
    model_deepspeed, optimizer_deepspeed, _, _ = dp.initialize(
        args=args,
        model=model,
        model_parameters=optimizer_grouped_parameters)

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    return model_deepspeed, optimizer_deepspeed, device
