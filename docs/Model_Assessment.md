# GitHub method assessment

The recommended engineering starting point is Random Forest point classification followed by DBSCAN instance grouping. This keeps the initial desktop installation small and allows the complete data and export workflow to be checked before adapting a specialised deep point-cloud model. It does not establish which method will perform best on the project's real targets.

The assessed official repositories provide reusable model implementations, but none establishes ready-to-use performance for the agreed Ratman project underwater target classes. LAZ reading, model adaptation and validated recognition are separate tasks.

| Project | Relevant capability | Integration considerations | Current status |
|---|---|---|---|
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | Random Forest, MLP and DBSCAN | CPU feature baseline; labelled training data required; clustering may merge touching targets | Integrated |
| [Pointcept](https://github.com/Pointcept/Pointcept) | Deep point-cloud backbones including PTv3; semantic configurations; PointGroup integration | Custom dataset, attribute channels, classes and original-point mapping; model-specific PyTorch/CUDA extensions; Linux/NVIDIA evaluation recommended first | Assessed, not integrated |
| [SoftGroup](https://github.com/thangvubk/SoftGroup) | Learning-based 3D instance segmentation | Semantic and instance annotations; sparse convolution and compiled extensions; custom dataset adaptation | Assessed, not integrated |
| [RandLA-Net](https://github.com/QingyongHu/RandLA-Net) | Large-scale semantic segmentation | Official README records an older Python/TensorFlow/CUDA environment; separate instance generation required | Reference candidate |
| [PointNet++](https://github.com/charlesq34/pointnet2) | Hierarchical point-set feature learning | Official TensorFlow implementation uses custom operators and requires custom-data preparation | Reference candidate |

PTv3 is a backbone, not an automatically deployable underwater instance detector. A semantic model still needs an instance method. SoftGroup targets instance segmentation, but its benchmark weights do not substitute for project-specific training or evaluation.

A published [Norway seagrass case study](https://www.yellowscan.com/knowledge/mapping-seagrass-ecosystems-in-norway-with-the-yellowscan-navigator-lidar/) reports a Random Forest refinement after CloudStation processing and manual cleaning. This is relevant evidence that the baseline is practical in a related bathymetric workflow; it does not demonstrate artificial-object detection performance.

The current MLP is a three-hidden-layer feature network. Random Forest is not deep learning, and the MLP baseline should not be represented as an end-to-end point-cloud architecture. If a specific deep-learning architecture is selected under REQ-11, its adaptation, installation, training and independent object-level validation remain required.

Official references were checked on 9 October 2026. Model repositories and their dependencies retain their own licences; review the licences of the exact versions selected for integration.
