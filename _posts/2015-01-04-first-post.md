---
layout: post
title: Rolling window with Pytorch
image: /img/hello_world.jpeg
tags:
  - rolling window
  - pytorch
  - deep learning
  - stock market
published: true
subtitle: Using pytorch for pratical things - rolling/sliding window
date: '2017-05-02'
---

I always make my neural network and deep learning stuffs using numpy from scratch ( this keep my mind always usefull ) and off couse for me, better for debug.
After heavly use _Tensor Flow_ and discover **Pytorch** I just love.
First because 95% off my models ( _actually not my but a implementation of many papers_ ) has been done from scratch ( and make my head explode many times ), see a framework make some things `easy` for you it's just like win in your birthday a box of cold beer from your girlfriend ( if you drink offcourse ).

So, when I start, first problem that I have was generate rolling windows ( _or slide window if you prefer_) just using pytorch (not with numpy), just with a simple line or couple of stride tricks but after read the docs I see how this was easy and pratical:

	# import torch
	import torch
	
	def pytorch_rolling_window(x, window_size, step_size=1):
        # unfold dimension to make our rolling window
        return x.unfold(0,window_size,step_size)
	
	# make a range sequence sample
	x = torch.range(1,20)
	
	# ie. window size of 5, step size of 1
	print(pytorch_rolling_window(x,5,1))
	
      1     2     3     4     5
      2     3     4     5     6
      3     4     5     6     7
      4     5     6     7     8
      5     6     7     8     9
      6     7     8     9    10
      7     8     9    10    11
      8     9    10    11    12
      9    10    11    12    13
     10    11    12    13    14
     11    12    13    14    15
     12    13    14    15    16
     13    14    15    16    17
     14    15    16    17    18
     15    16    17    18    19
     16    17    18    19    20
	[torch.FloatTensor of size 16x5]
    
    
That's it ;)
