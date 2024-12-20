# 2D Ground Penetrating Radar Simulation

Andrew Hoffmann ([@hoffmaao](https://github.com/hoffmaao)) converted the 2D radar simulation Matlab code from [this paper](https://doi.org/10.1016/j.cageo.2005.11.006) to Python. I've forked it to customize visualizations and tinker for my own purposes.

### Some of my goals are to:
* reproduce the figures from the paper
* compute shots in parallel
* make better animations / output of data cubes
* make an interactive earth model creator
* make a programmatic earth model creator

### Shot gather

![image](./figures/one_shot.png)

<img src="./figures/one_shot.png" width="600"/>

### Common offset gather (radar profile)

![image](./figures/common_offset.png)

<img src="./figures/common_offset.png" width="600"/>

### Wavefield animation

![animation](./figures/one_shot_wav.gif)

<img src="./figures/one_shot_wav.gif" width="600px">

### Shot gather animation

![animation](./figures/shots.gif)

<img src="./figures/shots.gif" width="600px">
