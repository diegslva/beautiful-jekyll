---
layout: post
published: true
title: A couple of tricks with Pytorch
subtitle: Little things with Pytorch that make a big headache disapper
image: /img/post_image_3.png
---
This post it's more like a _start_ of a recepie for things that I heavily use every day.
If you are like me that have platonic love for numpy and use for years since your born then inside Pytorch you don't have to mutch worry.
Numpy it's a awesome tool but if you need use **GPU** for you massive mega matrix multiplication so our beloved numpy can't help us. 
For that we have **Pytorch**! ( You can use [CuPy](http://docs.chainer.org/en/latest/cupy-reference/overview.html) too if your love for numpy its very maniac ( specialy for broadcast tricks ), but in this case for a better aproach it's best use [Chainer](https://github.com/pfnet/chainer/) for deep learning / Neural networks )


So, no more words, _Show me the code little buddy_!

_Where you see `Out[?]` it's because all examples was made with [IPython](https://ipython.org) if you don't know what is that, trust me, go to there and get use to it.
This is just a couple examples that I get it from [here](https://github.com/rougier/numpy-100/blob/master/100%20Numpy%20exercises.ipynb) but off course, using **Pytorch** this time_



    import torch

### 3. Create a null vector of size 10 (★☆☆)¶
    torch.zeros(1,10)
    Out[117]: 

        0     0     0     0     0     0     0     0     0     0
    [torch.FloatTensor of size 1x10]


### 4. How to find the memory size of any array (★☆☆)
    torch.zeros(1,10).size()
    Out[118]: torch.Size([1, 10])


### 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)
    Z = torch.zeros(1,10)
    Z[0,5] = 1

    Z
    Out[124]: 

        0     0     0     0     0     1     0     0     0     0
    [torch.FloatTensor of size 1x10]

### 7. Create a vector with values ranging from 10 to 49 (★☆☆)
    Z = torch.arange(10,50).view(1,-1)

    Z
    Out[128]: 


    Columns 0 to 12 
       10    11    12    13    14    15    16    17    18    19    20    21    22

    Columns 13 to 25 
       23    24    25    26    27    28    29    30    31    32    33    34    35

    Columns 26 to 38 
       36    37    38    39    40    41    42    43    44    45    46    47    48

    Columns 39 to 39 
       49
    [torch.FloatTensor of size 1x40]


### 8. Reverse a vector (first element becomes last) (★☆☆)¶
    # Using pytorch we don't have negative indices so we need to make little
    # trick using your own methods
    
    Z = torch.arange(20,30)
    Out[132]: 

       20
       21
       22
       23
       24
       25
       26
       27
       28
       29
      [torch.FloatTensor of size 10]
    
    # we create a range with inverse indices
    idx = torch.LongTensor([i for i in range(Z.size(0)-1,-1,-1)])
    
    Out[137]: 

     9
     8
     7
     6
     5
     4
     3
     2
     1
     0
    [torch.LongTensor of size 10]
    
    # done
    inverted_tensor = Z.index_select(0,idx)
    
    inverted_tensor
    Out[136]: 

     29
     28
     27
     26
     25
     24
     23
     22
     21
     20
    [torch.FloatTensor of size 10]


### 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)¶
    Z = torch.arange(0,9).view(3,3)

    Z
    Out[139]: 

     0  1  2
     3  4  5
     6  7  8
    [torch.FloatTensor of size 3x3]


### 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)¶
    Z = torch.LongTensor([1,2,0,0,4,0])

    non_zero_indices = torch.nonzero(Z)

    non_zero_indices
    Out[142]: 

     0
     1
     4
    [torch.LongTensor of size 3x1]



### 11. Create a 3x3 identity matrix (★☆☆)¶ -  “In×n”
    Z = torch.eye(3)

    Z
    Out[144]: 

     1  0  0
     0  1  0
     0  0  1
    [torch.FloatTensor of size 3x3]

### 12. Create a 3x3x3 array with random values (★☆☆)
    Z = torch.randn(3,3,3)

    Z
    Out[146]: 

    (0 ,.,.) = 
     -0.3989 -0.5808  0.2944
      0.5743  1.7362 -0.5612
     -0.2097  2.0039  0.0585

    (1 ,.,.) = 
     -1.9681 -1.2579 -0.4984
      0.1440 -0.0704  1.6027
     -1.2387 -0.7036 -0.8175

    (2 ,.,.) = 
      0.6146 -2.3633  1.9595
      0.6888 -0.7732  0.6254
     -0.7208 -0.4531 -0.0987
    [torch.FloatTensor of size 3x3x3]


### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
    Z = torch.randn(10,10)

    Zmin, Zmax = Z.min(), Z.max()

    Zmin, Zmax
    Out[149]: (-1.7579905986785889, 2.0110836029052734)
