---
layout: post
title: First post!
image: /img/hello_world.jpeg
tags:
  - random
  - exciting-stuff
published: true
---

## Rolling window with Pytorch

I always make my neural network and deep learning stuffs using numpy from scratch ( this keep my mind always usefull ) and off couse for me, better for debug
After heavly use Tensor Flow and discover Pytorch I just love.
First because 95% off my models ( actually not my but a implementation of many papers ) has been done from scratch ( and make my head explode many times ), see a framework make some things `easy` for you it's just like win your birthday a box of cold beer from your girlfriend ( if you drink offcourse ).

So, when I start, first problem that I have was generate rolling windows just using pytorch and after read the docs I see how this was easy:
`
# import torch
import torch

# make a range sequence sample
x = torch.range(1,20)

# unfold dimension to make our rolling window
# ie. window size of 5, step size of 1
x.unfold(0,5,1)
`

