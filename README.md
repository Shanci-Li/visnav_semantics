# TOPO-VNAV Semantics

  This repository shows the semester project of Sim-2-Real transfer of semantic segmentation networks for use in aerial applications.

  Due to vulnerabilities in GNSS/GPS Systems, on which autonomous systems rely for navigation and control, alternative methods are in demand for absolute large-scale navigation. The low cost, availability of cameras have made them a popular sensor for drones as a way to capture information on the surrounding landscape [1] (depth, semantics, feature and flow tracking). 
  State-of-the-art (ML) visual re-localisation based methods [2]–[4] show promising performance but typically focus on single-domain training. This poses a serious barrier for the adaptation in real-world scenarios as they require large datasets of high-quality real images within the desired area. Another lacking aspect is the lack of meaningful confidence bounds in the absolute pose estimation. A critical requirement for any downstream navigation filter/scheme.
  Previous projects at the TOPO lab [5]–[7] aimed at using synthetic data generated with the use of geospatial framework database Cesium [8]. The approach utilized at TOPO can best be summarized in two steps, the first being a coordinate regression by means of a deep neural network followed by the extraction of the pose by a PnP solver (Figure 1). 
  
![Image text](https://gitlab.epfl.ch/TOPO_MachineLearning/visnav_topo/visnav_semantics/-/blob/main/img/figure1.png)
  
The current architecture has shown the median performance of close to 6 m (absolute WSG84 position) and 2.5 deg (orientation) in synthetic validation datasets. This performance is significantly degraded when out of domain data is being used.

![Image text](https://gitlab.epfl.ch/TOPO_MachineLearning/visnav_topo/visnav_semantics/-/blob/main/img/figure2.png)

This semester project will seek to leverage on the novel concept of Mid Level Representations (MLR) [9], [10].
As shown in Figure 2 domain invariant MLR can be used as initial ‘filtering stage’ of the raw RGB image. Thus, the policy network downstream uses not the raw RGB image but rather the features from the MLR. A promising candidate for such MLR are the semantic segmentation networks. However, a limit in their performance is their domain invariance and ability to train them with minimum amount of real data. 

![Image text](https://gitlab.epfl.ch/TOPO_MachineLearning/visnav_topo/visnav_semantics/-/blob/main/img/figure3.png)

The purpose of this semester project will be to study state of the art semantic segmentation networks performance in and out of domain of training with provided by the project semantic segmentation labels available from Suisse TOPO as shown in Figure 3.
The task will be as following:
1)Perform a literature review of the latest semantic segmentation networks which can be trained on synthetic datasets (with a minimum number of real images) similar to [11].
2)To prepare a semantic segmentation labels using the available real and synthetic datasets for La Combalaz and EPFL. To split appropriately training, test and validation sets for both synthetic and real images.

3)To train the selected semantic segmentation algorithm on synthetic data and test in and out of domain.

4)To attempt sim-2-real transfer of the trained semantic segmentation algorithm and validate the improvement in out of domain training.

5)(Bonus task): To perform a benchmark study together with other students in the project the performance increase in absolute scene coordinate regression network trained with semantic segmentation network features as input. To compare the % improval of the Sim-2-Real transfer for the performance of the semantic segmentation features as MLR.
References:
[1]	A. Kendall et al., “End-to-End Learning of Geometry and Context for Deep Stereo Regression.”
[2]	E. Brachmann and C. Rother, “Expert Sample Consensus Applied to Camera Re-Localization.” Accessed: Sep. 28, 2020. [Online]. Available: http://vislearn.de.
[3]	E. Brachmann et al., “DSAC - Differentiable RANSAC for Camera Localization,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Jun. 2017, pp. 6684–6692, [Online]. Available: http://openaccess.thecvf.com/content_cvpr_2017/html/Brachmann_DSAC_-_Differentiable_CVPR_2017_paper.html.
[4]	E. Brachmann and C. Rother, “Visual Camera Re-Localization from RGB and RGB-D Images Using DSAC,” Feb. 2020, Accessed: Sep. 28, 2020. [Online]. Available: http://arxiv.org/abs/2002.12324.
[5]	Q. Yan, “OneShot Camera Pose Estimation for Synthetic RGB Images in Mountainous Terrain Scenes,” EPFL-TOPO, 2020.
[6]	Q. Yan, “Visual 6D Pose Estimation with Uncertainty Awareness,” EPFL-TOPO, 2021.
[7]	T. Shenker, “NO GPS? NO PROBLEM! TRANSFER LEARNING FOR DNNs USED IN AUTONOMOUS DRONE VISUAL NAVIGATION,” EPFL-TOPO, 2021.
[8]	“Cesium - Changing how the world views 3D.” https://cesium.com/ (accessed Sep. 09, 2020).
[9]	B. Chen et al., “Robust Policies via Mid-Level Visual Representations: An Experimental Study in Manipulation and Navigation,” arXiv, Nov. 2020, Accessed: Mar. 09, 2021. [Online]. Available: http://arxiv.org/abs/2011.06698.
[10]	A. Sax et al., “Learning to Navigate Using Mid-Level Visual Priors,” Accessed: Aug. 16, 2021. [Online]. Available: http://perceptual.actor.
[11]	J. Hoffman et al., “CyCADA: Cycle-Consistent Adversarial Domain Adaptation,” 2018.




## Install dependencies



## Setup datasets


## Dataset structure



## Getting started

