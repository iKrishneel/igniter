# Igniter
Igniter is simple configurable wrapper around the [PyTorch Ignite](https://docs.pytorch.org/ignite/index.html) library. 
Whist Ignite is powerful library that reduced the boilerplates when creating models, it is still not straight forward to use. 

The `Igniter` library builds on top of the `Ignite` and specifically tailored towards accelerating the model prototyping, training and testing.
For any model, the standard pipeline contains dataloader, model, training and testing routine. 
However, in most cases (if not all) the connection between these different essential components are very similar and therefore always implementing the pipeline for every model or data is cumbersome.

`Igniter` tries to simplify the process whereby the key focus is implementation of logics whist the underline connectivity between different components is handled by the library.
It depends on [hydra](https://hydra.cc/docs/intro/) for configuation.

## Install
To install the library

```bash
$ pip install igniter
```

## Idea
The key idea behind `Igniter` library is using registry. Registry is a global dictionary where different components can be added by simply using the decorators.
The `Registry` is modular by design, customizable, overwriteable and extendable.

#### Build-In Registry
There are 9 build in registry to handle various components of the machine learning pipeline

1. `Engine Registry` 
2. `Model Registry`
3. `Dataset Registry`
4. `Solver Registry`
5. `IO Registry`
6. `Function Registry`
7. `Transforms Registry`
8. `Event Handler Registry`
9. `Runner Registry`


## Creating Model
The model or the dataloader has to be created and then hooked up to igniter using the `decorators`

#### MNIST Model
Check out the full example script [here](https://github.com/iKrishneel/igniter/blob/main/example/mnist.py)

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, targets=None):
        device = self.conv1.weight.device
        x = x.to(device)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)

        if targets is not None:
            losses = self.losses(x, targets.to(device))
            if self.training:
                return losses
            return x, losses

        return x

    def losses(self, x, targets):
        return {'loss': F.nll_loss(x, targets)}
```

