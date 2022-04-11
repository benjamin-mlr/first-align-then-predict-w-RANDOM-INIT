import torch
import sys
sys.path.insert(0, "..")
from random_init.modeling_bert_random_init import BertModel


def test_random_init_model_layer_equal(model, model_w_random_init, layer_equal):
    """
    testing that the $layer_equal layer in the model after applying RANDOM-INIT to another layer is still equal to the original model
    :param model:
    :param model_w_random_init:
    :param layer_equal:
    :return:
    """

    # attention
    ## query
    assert torch.min(model.encoder.layer[layer_equal].attention.self.query.weight == model_w_random_init.encoder.layer[layer_equal].attention.self.query.weight) == torch.tensor(1)
    assert torch.min(model.encoder.layer[layer_equal].attention.self.query.bias == model_w_random_init.encoder.layer[layer_equal].attention.self.query.bias) == torch.tensor(1)
    ## key
    assert torch.min(model.encoder.layer[layer_equal].attention.self.key.weight == model_w_random_init.encoder.layer[layer_equal].attention.self.key.weight) == torch.tensor(1)
    assert torch.min(model.encoder.layer[layer_equal].attention.self.key.bias == model_w_random_init.encoder.layer[layer_equal].attention.self.key.bias) == torch.tensor(1)
    ## value
    assert torch.min(model.encoder.layer[layer_equal].attention.self.value.weight == model_w_random_init.encoder.layer[layer_equal].attention.self.value.weight) == torch.tensor(1)
    assert torch.min(model.encoder.layer[layer_equal].attention.self.value.bias == model_w_random_init.encoder.layer[layer_equal].attention.self.value.bias) == torch.tensor(1)
    ## attention output
    assert torch.min(model.encoder.layer[layer_equal].attention.output.dense.weight == model_w_random_init.encoder.layer[layer_equal].attention.output.dense.weight) == torch.tensor(1)
    assert torch.min(model.encoder.layer[layer_equal].attention.output.dense.bias == model_w_random_init.encoder.layer[layer_equal].attention.output.dense.bias) == torch.tensor(1)

    # intermediate
    assert torch.min(model.encoder.layer[layer_equal].intermediate.dense.weight == model_w_random_init.encoder.layer[layer_equal].intermediate.dense.weight) == torch.tensor(1)
    assert torch.min(model.encoder.layer[layer_equal].intermediate.dense.bias == model_w_random_init.encoder.layer[layer_equal].intermediate.dense.bias) == torch.tensor(1)

    # output
    assert torch.min(model.encoder.layer[layer_equal].output.dense.weight == model_w_random_init.encoder.layer[layer_equal].output.dense.weight) == torch.tensor(1)
    assert torch.min(model.encoder.layer[layer_equal].output.dense.bias == model_w_random_init.encoder.layer[layer_equal].output.dense.bias) == torch.tensor(1)

    assert torch.min(model.encoder.layer[layer_equal].output.LayerNorm.weight == model_w_random_init.encoder.layer[layer_equal].output.LayerNorm.weight) == torch.tensor(1)
    assert torch.min(model.encoder.layer[layer_equal].output.LayerNorm.bias == model_w_random_init.encoder.layer[layer_equal].output.LayerNorm.bias) == torch.tensor(1)


def test_random_init_model_layer_different(model, model_w_random_init, layer_random_init):
    """
    testing that the $layer_random_init layer after applying RANDOM-INIT to it is different than the original model
    :param model:
    :param model_w_random_init:
    :param layer_random_init:
    :return:
    """
    # attention
    ## query
    assert torch.min(model.encoder.layer[layer_random_init].attention.self.query.weight == model_w_random_init.encoder.layer[layer_random_init].attention.self.query.weight) == torch.tensor(0), f"no parameter are different: that is a bug {layer_random_init}"
    assert torch.min(model.encoder.layer[layer_random_init].attention.self.query.bias == model_w_random_init.encoder.layer[layer_random_init].attention.self.query.bias) == torch.tensor(0), f"failing for layer {layer_random_init}"
    ## key
    assert torch.min(model.encoder.layer[layer_random_init].attention.self.key.weight == model_w_random_init.encoder.layer[layer_random_init].attention.self.key.weight) == torch.tensor(0), f"failing for layer {layer_random_init}"
    assert torch.min(model.encoder.layer[layer_random_init].attention.self.key.bias == model_w_random_init.encoder.layer[layer_random_init].attention.self.key.bias) == torch.tensor(0), f"failing for layer {layer_random_init}"
    ## value
    assert torch.min(model.encoder.layer[layer_random_init].attention.self.value.weight == model_w_random_init.encoder.layer[layer_random_init].attention.self.value.weight) == torch.tensor(0), f"failing for layer {layer_random_init}"
    assert torch.min(model.encoder.layer[layer_random_init].attention.self.value.bias == model_w_random_init.encoder.layer[layer_random_init].attention.self.value.bias) == torch.tensor(0), f"failing for layer {layer_random_init}"
    ## attention output
    assert torch.min(model.encoder.layer[layer_random_init].attention.output.dense.weight == model_w_random_init.encoder.layer[layer_random_init].attention.output.dense.weight) == torch.tensor(0), f"failing for layer {layer_random_init}"
    assert torch.min(model.encoder.layer[layer_random_init].attention.output.dense.bias == model_w_random_init.encoder.layer[layer_random_init].attention.output.dense.bias) == torch.tensor(0), f"failing for layer {layer_random_init}"

    # intermediate
    assert torch.min(model.encoder.layer[layer_random_init].intermediate.dense.weight == model_w_random_init.encoder.layer[layer_random_init].intermediate.dense.weight) == torch.tensor(0), f"failing for layer {layer_random_init}"
    assert torch.min(model.encoder.layer[layer_random_init].intermediate.dense.bias == model_w_random_init.encoder.layer[layer_random_init].intermediate.dense.bias) == torch.tensor(0), f"failing for layer {layer_random_init}"

    # output
    assert torch.min(model.encoder.layer[layer_random_init].output.dense.weight == model_w_random_init.encoder.layer[layer_random_init].output.dense.weight) == torch.tensor(0), f"failing for layer {layer_random_init}"
    assert torch.min(model.encoder.layer[layer_random_init].output.dense.bias == model_w_random_init.encoder.layer[layer_random_init].output.dense.bias) == torch.tensor(0), f"failing for layer {layer_random_init}"

    assert torch.min(model.encoder.layer[layer_random_init].output.LayerNorm.weight == model_w_random_init.encoder.layer[layer_random_init].output.LayerNorm.weight) == torch.tensor(0), f"failing for layer {layer_random_init}"
    assert torch.min(model.encoder.layer[layer_random_init].output.LayerNorm.bias == model_w_random_init.encoder.layer[layer_random_init].output.LayerNorm.bias) == torch.tensor(0), f"failing for layer {layer_random_init}"


if __name__ == "__main__":

    mbert = BertModel.from_pretrained("bert-base-multilingual-cased")

    # TEST 1
    # test RANDOM-INIT applied to layer 0 to 1
    mbert_w_random_init = BertModel.from_pretrained("bert-base-multilingual-cased", random_init_layers=['bert.encoder.layer.[0-1]{1}.attention.*', 'bert.encoder.layer.[0-1]{1}.output.*', 'bert.encoder.layer.[0-1]{1}.intermediate.*'])

    layer_random_init = [0, 1]

    for layer_different in layer_random_init:
        test_random_init_model_layer_different(mbert, mbert_w_random_init, layer_different)
        print(f"Test succeeded: Layer {layer_different} has been randomly initialized")

    for layer_same in range(12):
        if layer_same not in layer_random_init:
            test_random_init_model_layer_equal(mbert, mbert_w_random_init, layer_same)
            print(f"Test succeeded: Layer {layer_same} is the same as in the original model")


    # TEST 2
    # test RANDOM-INIT applied to layer 10 to 11
    mbert_w_random_init = BertModel.from_pretrained("bert-base-multilingual-cased", random_init_layers=['bert.encoder.layer.[10]{2}.attention.*', 'bert.encoder.layer.[10]{2}.output.*', 'bert.encoder.layer.[10]{2}.intermediate.*'])

    layer_random_init = [10, 11]

    for layer_different in layer_random_init:
        test_random_init_model_layer_different(mbert, mbert_w_random_init, layer_different)
        print(f"Test succeeded: Layer {layer_different} has been randomly initialized")

    for layer_same in range(12):
        if layer_same not in layer_random_init:
            test_random_init_model_layer_equal(mbert, mbert_w_random_init, layer_same)
            print(f"Test succeeded: Layer {layer_same} is the same as in the original model")
