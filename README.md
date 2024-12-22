# Deep Segmented DMP Network (DSDNet)

Deep Segmented DMP Networks for Learning Discontinuous Motions  
by Edgar Anarossi, Hirotaka Tahara, Naoto Komeno, and Takamitsu Matsubara

## Abstract

Discontinuous motion, which is a motion composed of multiple continuous motions with sudden changes in direction or velocity in between, can be observed in state-aware robotic tasks. Such robotic tasks are often coordinated with sensor information such as images. In recent years, Dynamic Movement Primitives (DMP), a method for generating motor behaviors suitable for robotics, has garnered several deep learning-based improvements to allow associations between sensor information and DMP parameters. While the implementation of a deep learning framework does improve upon DMP's inability to directly associate with an input, we found that it has difficulty learning DMP parameters for complex motions that require a large number of basis functions to reconstruct. In this paper, we propose a novel deep learning network architecture called Deep Segmented DMP Network (DSDNet), which generates variable-length segmented motion by utilizing the combination of multiple DMP parameters predicting network architecture, double-stage decoder network, and number of segments predictor. The proposed method is evaluated on both artificial data (object cutting & pick-and-place) and real data (object cutting), where our proposed method achieves high generalization capability, task achievement, and data efficiency compared to previous methods for generating discontinuous long-horizon motions.

The paper can be accessed from the following links:
- [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10260679/)
- [arXiv](https://arxiv.org/pdf/2309.00320)

The models can be accessed in the `networks.py` file located in the `scripts/utils` directory.

## Introduction

This repository contains the implementation of the Deep Segmented DMP Network (DSDNet) for learning discontinuous motions in robotic tasks. The network architecture combines multiple DMP parameters predicting network architecture, a double-stage decoder network, and a number of segments predictor to generate variable-length segmented motions.

## Usage

To use the code, follow these steps:
1. Prepare your dataset and place it in the `data` directory.
2. Configure the training parameters in the `config.yaml` file.
3. Run the training script:
```bash
python train.py --config config.yaml
```

## Results

The results of the experiments can be found in the `results` directory. The key metrics and visualizations are provided to evaluate the performance of the proposed method.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## References

If you use this code in your research, please cite the following paper:
```
@inproceedings{anarossi2023deep,
  title={Deep segmented dmp networks for learning discontinuous motions},
  author={Anarossi, Edgar and Tahara, Hirotaka and Komeno, Naoto and Matsubara, Takamitsu},
  booktitle={2023 IEEE 19th International Conference on Automation Science and Engineering (CASE)},
  pages={1--7},
  year={2023},
  organization={IEEE}
}
```
