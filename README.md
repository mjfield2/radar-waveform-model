# 2D Ground Penetrating Radar Simulation

Andrew Hoffmann (@hoffmaao) converted the 2D radar simulation Matlab code from [this paper](https://www.sciencedirect.com/science/article/pii/S0098300405002621?casa_token=wbRlY4wIqoIAAAAA:yCFjd0qhQR6oV0vPRf1kLWQXJepc7GmzVVt8aIqN44Uio3zjoWd2OcKBIKXzb4vywFjAoHycuw) to Python. I've forked it to customize visualizations and tinker for my own purposes.

### Some of my goals are to:
* reproduce the figures from the paper
* compute shots in parallel
* make better animations / output of data cubes
* make an interactive earth model creator
* make a programmatic earth model creator

### Shot gather

![image](./figures/one.shot.png)

### Common offset gather (radar profile)

![image](./figures/common_offset.png)

### Wavefield animation

![](./one_shot_wav.mp4)
