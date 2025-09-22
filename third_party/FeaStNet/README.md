

# FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis

This is a Pytorch implementation of Feature-Steered Graph Convolutions (FeaStNet) for the task of dense shape correspondence on FAUST human dataset, as described in the paper:

Verma *et al*, [FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis](https://arxiv.org/abs/1706.05206) (CVPR 2018)

This implementation produces results better than those shown in the paper with the exactly same network architecture.

## Requirements
* [Pytorch](https://pytorch.org/) (1.3.0)
* [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) (1.3.0)

## FeaStNet

As a typical attention-based operator, FeaStNet learns a soft mapping from vertices to filter weights. The convolution is:
<p align="center"><img src="svgs/da0d4e5593311ecaab791ae9ffbc1fcb.svg" align=middle width=270.5142pt height=51.658694999999994pt/></p>

where assignment function <img src="svgs/1e6f5b1ba684ad7fce9ada1c076c2980.svg" align=middle width=255.260445pt height=27.852989999999977pt/>. Let <img src="svgs/3b2de8ebd8f32bc84779ecee22a9c6c4.svg" align=middle width=70.33718999999999pt height=19.10667000000001pt/>, which results int translation invariance of the weights in the feature space, giving much better performance.

We provide efficient Pytorch implementation of the translation-invariant version of this operator ``FeaStConv``. You should also be able to access this operator from the ``Pytorch Geometric`` Library.



## Run

```
python main.py
```

## Data

In order to use your own dataset, you can simply create a regular python list holding `torch_geometric.data.Data` objects and specify the following attributes:

- ``data.x``: Node feature matrix with shape ``[num_nodes, num_node_features]``
- ``data.edge_index``: Graph connectivity in COO format with shape ``[2, num_edges]`` and type ``torch.long``
- ``data.edge_attr``: Pesudo-coordinates with shape ``[num_edges, pesudo-coordinates-dim]``
- ``data.y``: Target to train against


## Cite

Please cite [this paper](https://arxiv.org/abs/1706.05206) if you use this code in your own work:

```
@inproceedings{verma2018feastnet,
  title={Feastnet: Feature-steered graph convolutions for 3d shape analysis},
  author={Verma, Nitika and Boyer, Edmond and Verbeek, Jakob},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2598--2606},
  year={2018}
}
```
