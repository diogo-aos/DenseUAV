import torch.nn as nn
from .Backbone.backbone import make_backbone
from .Head.head import make_head
import os
import torch

def ensure_correct_shape(x, oc: int):
    '''Ensure that the shape of tensor is (batch_size, whatever, output_channel).'''

    # Get the batch size
    bs = x.shape[0]

    # Check if the tensor is already in the correct shape
    if len(x.shape) == 3 and x.shape[2] == oc:
        return x

    # Find the index of the output channel dimension
    oc_index = None
    for i in range(1, len(x.shape)):
        if x.shape[i] == oc:
            oc_index = i
            break

    if oc_index is None:
        raise ValueError("Output channel dimension not found in tensor shape.")

    # Collapse all dimensions except batch_size and output_channel
    new_shape = (bs, -1, oc)

    # Permute dimensions to place output_channel at the end
    permute_dims = list(range(len(x.shape)))
    permute_dims.pop(oc_index)
    permute_dims.append(oc_index)

    # Reshape and permute the tensor
    x = x.permute(permute_dims).reshape(new_shape)

    return x


    


class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.backbone = make_backbone(opt)
        opt.in_planes = self.backbone.output_channel
        self.head = make_head(opt)
        self.opt = opt

    def forward(self, drone_image, satellite_image):
        oc = self.backbone.output_channel

        if drone_image is None:
            drone_res = None
        else:
            drone_features = self.backbone(drone_image)
            #drone_features = ensure_correct_shape(drone_features, oc)
            drone_res = self.head(drone_features)
        if satellite_image is None:
            satellite_res = None
        else:
            satellite_features = self.backbone(satellite_image)
            #satellite_features = ensure_correct_shape(satellite_features, oc)
            satellite_res = self.head(satellite_features)
        
        return drone_res,satellite_res
    
    def load_params(self, load_from):
        pretran_model = torch.load(load_from)
        model2_dict = self.state_dict()
        state_dict = {k: v for k, v in pretran_model.items() if k in model2_dict.keys() and v.size() == model2_dict[k].size()}
        model2_dict.update(state_dict)
        self.load_state_dict(model2_dict)


def make_model(opt):
    model = Model(opt)
    if os.path.exists(opt.load_from):
        model.load_params(opt.load_from)
    return model
