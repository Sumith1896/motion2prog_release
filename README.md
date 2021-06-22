# Hierarchical Motion Understanding via Motion Programs (CVPR 2021)

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2104.11216)

This repository contains the official implementation of:

**Hierarchical Motion Understanding via Motion Programs**

[Full paper](https://arxiv.org/abs/2104.11216) | [short talk](https://www.youtube.com/watch?v=EKP2BIRlaXQ) | [long talk](https://www.youtube.com/watch?v=OpyY-s0LKAs) | [Project website](https://sumith1896.github.io/motion2prog/) 


## Running motion2prog

**0. We start with video file and first prepare the input data**

```sh
$ ffmpeg -i ${video_dir}/video.mp4 ${video_dir}/frames/%05d.jpg
$ python AlphaPose/scripts/demo_inference.py \
    --cfg AlphaPose/pretrained_models/256x192_res50_lr1e-3_1x.yaml \
    --checkpoint AlphaPose/pretrained_models/halpe26_fast_res50_256x192.pth \
    --indir ${video_dir}/frames --outdir ${video_dir}/pose_mpii_track \
    --pose_track --showbox --flip --qsize 256
$ mv ${video_dir}/pose_mpii_track/alphapose-results.json \
    ${video_dir}/alphapose-results-halpe26-posetrack.json
```

We packaged a demo video with necessary inputs for quickly testing our code

```sh
$ wget https://sumith1896.github.io/motion2prog/static/demo.zip
$ mv demo.zip data/  && cd data/ && unzip demo.zip && cd ..
```

- We need 2D pose detection results & extracted frames of video (for visualization)

- We support loading from different pose detector formats in the `load` function in `lkeypoints.py`.

- We used `AlphaPose` with the above commands for all pose detection results.

### Run motion program synthesis pipeline
**1. With the data prepared, you can run the synthesis with the following command:**

```sh
$ python fit.py -d data/demo/276_reg -k coco -a -x -c -p 1 -w 20 --no-acc \
--stat-thres 5 --span-thres 5 --cores 9 -r 1600 -o ./visualization/static/data/demo
```

- The various options and their descriptions are explained in the `fit.py` file.

- The results can be found under `./visualization/static/data/demo`.

### Visualizing the synthesized programs
**2. We package a visualization server for visualizing the generated programs**
```sh
$ cd visualization/
$ bash deploy.sh p
```

- Open the directed the webpage and browse the results interactively.


## Citations
If you find our code or paper useful to your research, please consider citing:

```bibtex
@inproceedings{motion2prog2021,
    Author = {Sumith Kulal and Jiayuan Mao and Alex Aiken and Jiajun Wu},
    Title = {Hierarchical Motion Understanding via Motion Programs},
    booktitle={CVPR},
    year={2021},
}
```
