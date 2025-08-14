<div align="center">

<h1>Style4D-Bench: A Benchmark Suite for 4D Stylization </h1>






### :baby_chick: [Project](https://sparsenerf.github.io/) | [YouTube](https://youtu.be/Tf6QnksXFxQ) | [arXiv]() 
<div>

</div>
<tr>
    <img src="assets/result.gif" width="90%"/>
</tr>
</div>



>:baby_chick: **Abstract:** We introduce Style4D-Bench, the first benchmark suite specifically designed for 4D stylization, with the goal of standardizing evaluation and facilitating progress in this emerging area. Style4D-Bench comprises: 1) a strong baseline that make an initial attempt for 4D stylization, 2) a comprehensive evaluation protocol measuring spatial fidelity, temporal coherence, and multi-view consistency through both perceptual and quantitative metrics, and 3) a curated collection of high-resolution dynamic 4D scenes with diverse motions and complex backgrounds. To establish a strong baseline, we present Style4D, a novel framework built upon 4D Gaussian Splatting. It consists of three key components: a basic 4DGS scene representation to capture reliable geometry, a Style Gaussian Representation that leverages lightweight per-Gaussian MLPs for temporally and spatially aware appearance control, and a Holistic Geometry-Preserved Style Transfer module designed to enhance spatio-temporal consistency via contrastive coherence learning and structural content preservation. Extensive experiments on Style4D-Bench demonstrate that Style4D achieves state-of-the-art performance in 4D stylization, producing fine-grained stylistic details with stable temporal dynamics and consistent multi-view rendering. We expect Style4D-Bench to become a valuable resource for benchmarking and advancing research in stylized rendering of dynamic 3D scenes. 
<div align="center">
<tr>
    <img src="assets/4DGS_model.png" width="90%"/>
</tr>
</div>

>:baby_chick: **Framework Overview:** Style4D consists of three key components, a basic 4DGS representation, a Style Gaussian Representation, and a Holistic Geometry-preserved Style Transfer. We first train a basic 4DGS representation with the content image to obtain 4D scene geometry. Then we propose a new Style Gaussian Representation for 4D stylization. We also introduce a Holistic Geometry-preserved Style Transfer module to improve consistency and quality of stylization.

<div align="center">
<tr>
    <img src="assets/metric.png" width="90%"/>
</tr>
</div>

>:baby_chick: **Evaluation Metrics:** Quantitative comparisons of our proposed Style4D against state-of-the-art methods on Style4D-Bench.





##  Citation

If you find this useful for your research, please cite the our paper.


