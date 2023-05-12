import TensionFlow as tf
import torch
import numpy as np

#testing all gradients by comparing with pytorch


#test addition
def test_add():
    for i in range(100):
        a = np.random.rand(10,10)
        b = np.random.rand(10,10)
        a = tf.Neuron(a)
        b = tf.Neuron(b)
        c = a+b
        c.sum().backward()
        a1 = torch.tensor(a.value, requires_grad=True)
        b1 = torch.tensor(b.value, requires_grad=True)
        c1 = a1+b1
        c1.sum().backward()
        assert np.allclose(a.grad, a1.grad)
        assert np.allclose(b.grad, b1.grad)
        assert np.allclose(c.value, c1.detach().numpy())

#test multiplication
def test_mul():
    for i in range(100):
        a = np.random.rand(10,10)
        b = np.random.rand(10,10)
        a = tf.Neuron(a)
        b = tf.Neuron(b)
        c = a*b
        c.sum().backward()
        a1 = torch.tensor(a.value, requires_grad=True)
        b1 = torch.tensor(b.value, requires_grad=True)
        c1 = a1*b1
        c1.sum().backward()
        assert np.allclose(a.grad, a1.grad)
        assert np.allclose(b.grad, b1.grad)
        assert np.allclose(c.value, c1.detach().numpy())

#test matrix multiplication
def test_matmul():
    for i in range(100):
        a = np.random.rand(10,10)
        b = np.random.rand(10,10)
        a = tf.Neuron(a)
        b = tf.Neuron(b)
        c = a@b
        c.sum().backward()
        a1 = torch.tensor(a.value, requires_grad=True)
        b1 = torch.tensor(b.value, requires_grad=True)
        c1 = a1@b1
        c1.sum().backward()
        assert np.allclose(a.grad, a1.grad)
        assert np.allclose(b.grad, b1.grad)
        assert np.allclose(c.value, c1.detach().numpy())

#test transpose
def test_transpose():
    for i in range(100):
        a = np.random.rand(10,10)
        a = tf.Neuron(a)
        c = a.T
        c.sum().backward()
        a1 = torch.tensor(a.value, requires_grad=True)
        c1 = a1.T
        c1.sum().backward()
        assert np.allclose(a.grad, a1.grad)
        assert np.allclose(c.value, c1.detach().numpy())

#test relu
def test_relu():
    for i in range(100):
        a = np.random.rand(10,10)
        a = tf.Neuron(a)
        c = tf.ReLU(a)
        c.sum().backward()
        a1 = torch.tensor(a.value, requires_grad=True)
        c1 = a1.relu()
        c1.sum().backward()
        assert np.allclose(a.grad, a1.grad)
        assert np.allclose(c.value, c1.detach().numpy())

#test div
def test_div():
    for i in range(100):
        a = np.random.rand(10,10)
        b = np.random.rand(10,10)
        a = tf.Neuron(a)
        b = tf.Neuron(b)
        c = a/b
        c.sum().backward()
        a1 = torch.tensor(a.value, requires_grad=True)
        b1 = torch.tensor(b.value, requires_grad=True)
        c1 = a1/b1
        c1.sum().backward()
        assert np.allclose(a.grad, a1.grad)
        assert np.allclose(b.grad, b1.grad)
        assert np.allclose(c.value, c1.detach().numpy())

#test exp
def test_exp():
    for i in range(100):
        a = np.random.rand(10,10)
        a = tf.Neuron(a)
        c = a.exp()
        c.sum().backward()
        a1 = torch.tensor(a.value, requires_grad=True)
        c1 = a1.exp()
        c1.sum().backward()
        assert np.allclose(a.grad, a1.grad)
        assert np.allclose(c.value, c1.detach().numpy())

#test broadcast
def test_broadcast():
    for i in range(100):
        a = np.random.rand(10,10)
        b = np.random.rand(1,10)
        a = tf.Neuron(a)
        b = tf.Neuron(b)
        c = a+b.broadcast(10)
        c.sum().backward()
        a1 = torch.tensor(a.value, requires_grad=True)
        b1 = torch.tensor(b.value, requires_grad=True)
        c1 = a1+b1
        c1.sum().backward()
        assert np.allclose(a.grad, a1.grad)
        assert np.allclose(b.grad, b1.grad)
        assert np.allclose(c.value, c1.detach().numpy())
#test sum
def test_sum():
    for i in range(100):
        a = np.random.rand(10,10)
        a = tf.Neuron(a)
        c = a.sum(dim=0)
        c.sum().backward()
        a1 = torch.tensor(a.value, requires_grad=True)
        c1 = a1.sum(dim=0)
        c1.sum().backward()
        assert np.allclose(a.grad, a1.grad)
        assert np.allclose(c.value, c1.detach().numpy())
def test_max():
    for i in range(100):
        a = np.random.rand(10,10)
        a = tf.Neuron(a)
        c = a.max(dim=0)
        c.sum().backward()
        a1 = torch.tensor(a.value, requires_grad=True)
        c1,_ = a1.max(dim=0)
        c1.sum().backward()
        assert np.allclose(c.value, c1.detach().numpy())
        assert np.allclose(a.grad, a1.grad)

#test softmax
def test_softmax():
    for i in range(100):
        a = np.random.rand(10,10)
        a = tf.Neuron(a)
        c = tf.Softmax(a, dim=1)
        c.sum(None).backward()
        a1 = torch.tensor(a.value, requires_grad=True)
        c1 = torch.softmax(a1, dim=1)
        c1.sum().backward()
        assert np.allclose(a.grad, a1.grad), f"incorrect answer: {a.grad} {a1.grad}"
        assert np.allclose(c.value, c1.detach().numpy())

if __name__ == "__main__":
    test_add()
    test_mul()
    test_matmul()
    test_relu()
    test_div()
    test_exp()
    test_broadcast()
    test_max()

    test_softmax()
    print("All tests passed")