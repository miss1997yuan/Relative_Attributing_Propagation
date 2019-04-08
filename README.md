# Interpreting Deep Neural Networks - Relative Attributing Propagation
Relative attributing propagation (RAP) decomposes the output predictions of DNNs with a perspective that precisely separates the positive and negative attributions.

Detail description of this method is provided in our paper https://arxiv.org/pdf/1904.00605.pdf.

This code provides a simple implementation of RAP.
For implementing other explaining methods [Layerwise relevance propagation(LRP), Deep taylor decomposition(DTD)], we followed the tutorial of http://heatmapping.org and https://github.com/VigneshSrinivasan10/interprettensor.

![Alt text](/result.png)

# Requirements
	tensorflow >= 1.0.0
	python >= 3
	matplotlib >= 1.3.1
	scikit-image > 0.11.3

# Run
Before running this code, vgg-16 network is required.
Please download the model from https://drive.google.com/file/d/1NsX4m_Mp9j0tU-oyM6nE-CxHPl61Ylhe/view?usp=sharing and save it as vgg16.npy in directory.

	python imagenet_convolutional.py



# Paper Citation
When using this code, please cite our paper.

	@misc{nam2019relative,
			title={Relative Attributing Propagation: Interpreting the Comparative Contributions of Individual Units in Deep Neural Networks},
			author={Woo-Jeoung Nam and Jaesik Choi and Seong-Whan Lee},
			year={2019},
			eprint={1904.00605},
			archivePrefix={arXiv},
			primaryClass={cs.CV}
	}
