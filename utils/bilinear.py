import torch
import torch.nn.functional as F


"""
Functions for performing differentiable bilinear cropping of images, for use in the object discriminator
Modified from https://github.com/google/sg2im/blob/master/sg2im/bilinear.py
"""


def stn(image, transformation_matrix, size):
    grid = torch.nn.functional.affine_grid(transformation_matrix, torch.Size(size))
    out_image = torch.nn.functional.grid_sample(image, grid)

    return out_image


def crop_bbox(feats, bbox, HH, WW=None, backend='cudnn'):
    """
    Take differentiable crops of feats specified by bbox.
    Inputs:
        - feats: Tensor of shape (N, C, H, W)
        - bbox: Bounding box coordinates of shape (N, 4) in the format
        [x0, y0, x1, y1] in the [0, 1] coordinate space.
        - HH, WW: Size of the output crops.
    Returns:
        - crops: Tensor of shape (N, C, HH, WW) where crops[i] is the portion of
        feats[i] specified by bbox[i], reshaped to (HH, WW) using bilinear sampling.
    """
    N = feats.size(0)
    assert bbox.size(0) == N
    assert bbox.size(1) == 4
    if WW is None: WW = HH
    # if backend == 'cudnn':
    # Change box from [0, 1] to [-1, 1] coordinate system
    #    bbox = 2 * bbox - 1
    x0, y0 = 2 * bbox[:, 0] - 1, 2 * bbox[:, 1] - 1
    x1, y1 = 2 * (bbox[:, 2] + bbox[:, 0]) - 1, 2 * (bbox[:, 3] + bbox[:, 1]) - 1

    X = tensor_linspace(x0, x1, steps=WW).view(N, 1, WW).expand(N, HH, WW).cuda(device=feats.device)
    Y = tensor_linspace(y0, y1, steps=HH).view(N, HH, 1).expand(N, HH, WW).cuda(device=feats.device)

    if backend == 'jj':
        return bilinear_sample(feats, X, Y)
    elif backend == 'cudnn':
        grid = torch.stack([X, Y], dim=3)
        return F.grid_sample(feats, grid)


def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
        - start: Tensor of any shape
        - end: Tensor of the same shape as start
        - steps: Integer
    Returns:
        - out: Tensor of shape start.size() + (steps,), such that
        out.select(-1, 0) == start, out.select(-1, -1) == end,
        and the other elements of out linearly interpolate between
        start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def bilinear_sample(feats, X, Y):
    """
    Perform bilinear sampling on the features in feats using the sampling grid
    given by X and Y.
    Inputs:
        - feats: Tensor holding input feature map, of shape (N, C, H, W)
        - X, Y: Tensors holding x and y coordinates of the sampling
        grids; both have shape shape (N, HH, WW) and have elements in the range [0, 1].
    Returns:
        - out: Tensor of shape (B, C, HH, WW) where out[i] is computed
        by sampling from feats[idx[i]] using the sampling grid (X[i], Y[i]).
    """
    N, C, H, W = feats.size()
    assert X.size() == Y.size()
    assert X.size(0) == N
    _, HH, WW = X.size()

    X = X.mul(W)
    Y = Y.mul(H)

    # Get the x and y coordinates for the four samples
    x0 = X.floor().clamp(min=0, max=W-1)
    x1 = (x0 + 1).clamp(min=0, max=W-1)
    y0 = Y.floor().clamp(min=0, max=H-1)
    y1 = (y0 + 1).clamp(min=0, max=H-1)

    # In numpy we could do something like feats[i, :, y0, x0] to pull out
    # the elements of feats at coordinates y0 and x0, but PyTorch doesn't
    # yet support this style of indexing. Instead we have to use the gather
    # method, which only allows us to index along one dimension at a time;
    # therefore we will collapse the features (BB, C, H, W) into (BB, C, H * W)
    # and index along the last dimension. Below we generate linear indices into
    # the collapsed last dimension for each of the four combinations we need.
    y0x0_idx = (W * y0 + x0).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y1x0_idx = (W * y1 + x0).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y0x1_idx = (W * y0 + x1).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y1x1_idx = (W * y1 + x1).view(N, 1, HH * WW).expand(N, C, HH * WW)

    # Actually use gather to pull out the values from feats corresponding
    # to our four samples, then reshape them to (BB, C, HH, WW)
    feats_flat = feats.view(N, C, H * W)
    v1 = feats_flat.gather(2, y0x0_idx.long()).view(N, C, HH, WW)
    v2 = feats_flat.gather(2, y1x0_idx.long()).view(N, C, HH, WW)
    v3 = feats_flat.gather(2, y0x1_idx.long()).view(N, C, HH, WW)
    v4 = feats_flat.gather(2, y1x1_idx.long()).view(N, C, HH, WW)

    # Compute the weights for the four samples
    w1 = ((x1 - X) * (y1 - Y)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w2 = ((x1 - X) * (Y - y0)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w3 = ((X - x0) * (y1 - Y)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w4 = ((X - x0) * (Y - y0)).view(N, 1, HH, WW).expand(N, C, HH, WW)

    # Multiply the samples by the weights to give our interpolated results.
    out = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return out


def masks_to_layout(boxes, masks, H, W=None):
    """
    Inputs:
        - boxes: Tensor of shape (b, num_o, 4) giving bounding boxes in the format
            [x0, y0, x1, y1] in the [0, 1] coordinate space
        - masks: Tensor of shape (b, num_o, M, M) giving binary masks for each object
        - H, W: Size of the output image.
    Returns:
        - out: Tensor of shape (N, num_o, H, W)
    """
    b, num_o, _ = boxes.size()
    M = masks.size(2)
    assert masks.size() == (b, num_o, M, M)
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes.view(b*num_o, -1), H, W).float().cuda(device=masks.device)

    img_in = masks.float().view(b*num_o, 1, M, M)
    sampled = F.grid_sample(img_in, grid, mode='bilinear')

    return sampled.view(b, num_o, H, W)


def _boxes_to_grid(boxes, H, W):
    """
    Input:
    - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
      format in the [0, 1] coordinate space
    - H, W: Scalars giving size of output
    Returns:
    - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
    """
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    ww, hh = boxes[:, 2], boxes[:, 3]

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww  # (O, 1, W)
    Y = (Y - y0) / hh  # (O, H, 1)

    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)

    return grid


def compute_transformation_matrix(bbox):
    x, y = bbox[:, 0], bbox[:, 1]
    w, h = bbox[:, 2], bbox[:, 3]

    scale_x = w
    scale_y = h

    t_x = 2 * ((x + 0.5 * w) - 0.5)
    t_y = 2 * ((y + 0.5 * h) - 0.5)

    zeros = torch.cuda.FloatTensor(bbox.shape[0],1).fill_(0)

    transformation_matrix = torch.cat([scale_x.unsqueeze(-1), zeros, t_x.unsqueeze(-1),
                                       zeros, scale_y.unsqueeze(-1), t_y.unsqueeze(-1)], 1).view(-1, 2, 3)

    return transformation_matrix
