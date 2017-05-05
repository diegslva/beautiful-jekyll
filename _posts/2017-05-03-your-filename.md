---
layout: post
published: true
title: 'Deep Learning with Pytorch, a Simple Classifier'
image: /img/post_image_2.jpg
subtitle: A simple but not trivial classifier with Pytorch
---
## Simple Classifier using Pytorch

**Pytorch** it's for me a great framework in this days. A great comunity that help you if you need any time and with fast aswers for you.
BTW, check out here [https://discuss.pytorch.org/latest](https://discuss.pytorch.org/latest) if you start with pytorch or even if you have a lot of problem with your beloved tensors!

So, in this post I just want to show you a simple but no trivial classifier with a toy dataset. For me I always want a example with the dataset include for fast try! 
I don't know you but, `paste - code - running` without any trouble it's always a real pleasure ( see a `stack strace error` or whatever lib `not found` definetely is not)

In this example we try training our classifier accross a multi-class problem. Obviouly a just made up with random numbers and for our surprise they converge like magic.
Using pytorch you don't need to worry about `backpropagation`. If you don't no yet what is that, i really want you try look here [http://cs231n.github.io/optimization-2/#staged](http://cs231n.github.io/optimization-2/#staged) for a beatifull understand what is going on behind the scenes of pytorch.

This post that is not intend to teach about `what is` and more like `run it`. 
Probably I'll try post more articles with this intend!
So, just copy, paste and run using python and see pytorch make your magic.

**Requirements**
_I assume that you know what is a framework and you have it installed. If you don't, just go to http://pytorch.org/ and follow the easy steps ( actually is just one ;) )_



    import torch
    import torch.nn as nn
    from torch import np
    import torch.optim as optim
    from torch.autograd import Variable

    # Hot encode our classes
    # (1, 0, 0)  target labels 0
    # (0, 1, 0)  target labels 1
    # (0, 0, 1)  target labels 2
    train = []
    labels = []

    # we just made up our random training dataset
    # In this case we generate 10 thousands of trainnig samples
    for i in range(10000):
        category = (np.random.choice([0, 1]), np.random.choice([0, 1]))
        if category == (1, 0):
            train.append([np.random.uniform(0.1, 1), 0])
            labels.append([1, 0, 1])
        if category == (0, 1):
            train.append([0, np.random.uniform(0.1, 1)])
            labels.append([0, 1, 0])
        if category == (0, 0):
            train.append([np.random.uniform(0.1, 1), np.random.uniform(0.1, 1)])
            labels.append([0, 0, 1])

    # A simple but not trivial linear classifier
    class SimpleClassifier(nn.Module):    
        def __init__(self, nlabel):
            super(SimpleClassifier, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, nlabel),
            )
        #end __init__

        def forward(self, input):
            return self.main(input)
        #end forward
    #end SimpleClassifier


    # How many class!?
    number_of_class = len(labels[0]) # 3
    classifier = SimpleClassifier(number_of_class)

    # Use your favorite optimizer
    # In this case we will use Adam ( SGD will be fine here too)
    # our learning rate is 0.001
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    # Loss function 
    # In this case we use MultiLabelSoftMarginLoss because
    # our example its a multi-class problem
    # (Actually, is the same that a sigmoid_activation + binary_cross_entropy)
    # again, you dont need to worry about that for this example
    criterion = nn.MultiLabelSoftMarginLoss()

    epochs = 2
    for epoch in range(epochs):
        losses = []
        for i, sample in enumerate(train):
            # wrapper inside Variables our sample train
            # this way all operations will be saved for
            # autograd operations
            input_var = Variable(torch.FloatTensor(sample)).view(1, -1)
            # wrapper inside Variables our target
            target_var = Variable(torch.FloatTensor(labels[i])).view(1, -1)
            # get our result
            output = classifier(input_var)
            # calculate our loss
            loss = criterion(output, target_var)
            # zero gradients here or pytorch will accumulate
            optimizer.zero_grad()
            # using autograd technique, just call backward and done!
            loss.backward()
            # here we call our optimize, this is the same 
            # to something like this: w -= lr * (gradients)
            # pytorch make this easy so you dont need calculate then 
            # using your little fingers
            optimizer.step()
            # just save our average loss for display
            losses.append(loss.data.mean())
            print("[{}/{}] Loss: {}".format(epoch+1, epochs, np.mean(losses).round(3)))
