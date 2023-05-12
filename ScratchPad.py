import TensionFlow as tf
import torch
import numpy as np


a = np.random.rand(2,2)
a = tf.Neuron(a)
exp = a.exp()
sum_exp = exp.sum(0)
c = exp / sum_exp.broadcast(a.shape()[0])

c.sum(None).backward()
a1 = torch.tensor(a.value, requires_grad=True)
c1 = torch.nn.functional.softmax(a1, dim=0)
c1.sum().backward()

assert np.allclose(c1.detach().numpy(), c.value), "incorrect answer"
print(a)
print(a1.grad, 'Tension grad: ',  a.grad)