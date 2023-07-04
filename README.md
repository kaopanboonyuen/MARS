# MARS

MARS: Mask Attention Refinement with Sequential Quadtree Nodes for Car Damage Instance Segmentation

```
Evaluating car damages from misfortune is critical to the car insurance industry. However, the accuracy is still insufficient for real-world applications since the deep learning network is not designed for car damage images as inputs, and its segmented masks are still very coarse. This paper presents MARS (Mask Attention Refinement with Sequential quadtree nodes) for car damage instance segmentation. Our MARS represents self-attention mechanisms to draw global dependencies between the sequential quadtree nodes layer and quadtree transformer to recalibrate channel weights and predict highly accurate instance masks. Our extensive experiments demonstrate that MARS outperforms state-of-the-art (SOTA) instance segmentation methods on three popular benchmarks such as Mask R-CNN [9], PointRend [14], and Mask Transfiner [13], by a large margin of +1.3 maskAP-based R50-FPN backbone and +2.3 maskAP-based R101-FPN backbone on Thai car-damage dataset.
```

Our demos are available at https://www.marssolution.io/

## Reference:
```
@article{panboonyuen2023mars,
  title={MARS: Mask Attention Refinement with Sequential Quadtree Nodes for Car Damage Instance Segmentation},
  author={Panboonyuen, Teerapong and Nithisopa, Naphat and Pienroj, Panin and Jirachuphun, Laphonchai and Watthanasirikrit, Chaiwasut and Pornwiriyakul, Naruepon},
  journal={arXiv preprint arXiv:2305.04743},
  year={2023}
}
```

![](https://github.com/kaopanboonyuen/MARS/raw/main/img/MARS01.png)

![](https://github.com/kaopanboonyuen/MARS/raw/main/img/MARS02.png)

![](https://github.com/kaopanboonyuen/MARS/raw/main/img/MARS03.png)

![](https://github.com/kaopanboonyuen/MARS/raw/main/img/MARS04.png)