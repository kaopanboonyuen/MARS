# MARS: Mask Attention Refinement with Sequential Quadtree Nodes for Car Damage Instance Segmentation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Welcome to the official repository for **MARS**—a cutting-edge deep learning model designed for car damage instance segmentation. MARS leverages advanced self-attention mechanisms within sequential quadtree nodes to produce highly accurate segmentation masks, offering a significant improvement over state-of-the-art methods like Mask R-CNN, PointRend, and Mask Transfiner.

## Authors
Teerapong Panboonyuen (also known as Kao Panboonyuen)

## Project Overview

Accurately assessing car damage is crucial for the insurance industry, yet traditional deep learning models struggle with fine segmentation and image complexity. MARS (Mask Attention Refinement with Sequential Quadtree Nodes) introduces a novel approach, recalibrating channel weights using a quadtree transformer to enhance segmentation precision. Our model has been rigorously tested and has achieved notable performance gains:

- **+1.3 maskAP** improvement using the R50-FPN backbone
- **+2.3 maskAP** improvement with the R101-FPN backbone on the Thai car-damage dataset

MARS was presented at the **International Conference on Image Analysis and Processing 2023 (ICIAP 2023)** in Udine, Italy.

![](img/featured.png)

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.1+
- Other dependencies listed in `requirements.txt`

### Step-by-Step Guide

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kaopanboonyuen/MARS.git
   cd MARS
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv mars-env
   source mars-env/bin/activate  # On Windows use `mars-env\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the datasets**:
   - **Public Dataset:** [Download here](https://drive.google.com/file/d/1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx/view) and place it in the `data/` directory.
   - **Private Dataset:** Access is restricted due to licensing agreements.

## How to Run

1. **Train the Model**:
   ```bash
   python train.py --config configs/mars_config.yaml
   ```

2. **Evaluate the Model**:
   ```bash
   python evaluate.py --checkpoint checkpoints/mars_best_model.pth --data data/test/
   ```

3. **Run Inference**:
   ```bash
   python inference.py --image_path images/sample.jpg --output_dir results/
   ```

## Demos

Explore our live demos and see MARS in action: [GitHub Pages](https://kaopanboonyuen.github.io/MARS)

## Datasets

We trained our models on both public and private datasets:

- **Public Dataset:** Download [here](https://drive.google.com/file/d/1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx/view)
- **Private Dataset:** Access to our private dataset is restricted due to licensing agreements with **THAIVIVAT INSURANCE PCL.**, a partner in our tech startup, MARS.

## Citation

If you find our work useful in your research or development, please cite:

```bibtex
@inproceedings{panboonyuen2023mars,
  title={MARS: Mask Attention Refinement with Sequential Quadtree Nodes for Car Damage Instance Segmentation},
  author={Panboonyuen, Teerapong and Nithisopa, Naphat and Pienroj, Panin and Jirachuphun, Laphonchai and Watthanasirikrit, Chaiwasut and Pornwiriyakul, Naruepon},
  booktitle={International Conference on Image Analysis and Processing},
  pages={28--38},
  year={2023},
  organization={Springer}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or collaboration opportunities, feel free to reach out to me:

- **Author:** [Teerapong Panboonyuen (Kao Panboonyuen)](https://kaopanboonyuen.github.io)
- **Email:** panboonyuen.kao@gmail.com
- **MARS:** https://www.marssolution.io

![](img/MARS01.png)
![](img/MARS_001.png)
![](img/MARS_002.png)
![](img/MARS_003.png)
![](img/MARS_005.png)