import numpy as np

#making computation graph

class Neuron:
    def __init__(self, value):
        #value
        #local derivative
        self.value = value
        self.grad = None
        self.children = []
        self.is_pointer = False
        
        
    def __repr__(self):
        return str(f'value: {self.value}, grad: {self.grad}')
    
    def _handle_back_add(self,other_neuron=None):
        if self.grad is not None:
            new_neuron = Neuron(self.value)
            new_neuron.is_pointer = True
            new_neuron.children = [self]
            self = new_neuron
        elif other_neuron is not None and other_neuron.grad is not None:
            new_neuron = Neuron(other_neuron.value)
            new_neuron.is_pointer = True
            new_neuron.children = [other_neuron]
            other_neuron = new_neuron
        elif other_neuron is not None and other_neuron is self:
            new_neuron = Neuron(self.value)
            new_neuron.is_pointer = True
            new_neuron.children = [other_neuron]
            other_neuron = new_neuron
        return self,other_neuron
    
    def __mul__(self, other_neuron):
        #if not a neuron then create a neuron
        if not isinstance(other_neuron, Neuron):
            other_neuron = Neuron(other_neuron)
        self,other_neuron = self._handle_back_add(other_neuron)
        new_neuron = Neuron(self.value * other_neuron.value)
        self.grad = other_neuron.value
        other_neuron.grad = self.value
        new_neuron.children = [self,other_neuron]
            
        return new_neuron
        
    def __add__(self, other_neuron):
        if not isinstance(other_neuron, Neuron):
            other_neuron = Neuron(other_neuron)
        self,other_neuron = self._handle_back_add(other_neuron)
        new_neuron = Neuron(self.value + other_neuron.value)
        self.grad = 1
        other_neuron.grad = 1
        new_neuron.children = [self,other_neuron]
        return new_neuron
    
    #setting right add and mul to mul and add
    __radd__ = __add__
    __rmul__ = __mul__
    
    def __neg__(self):
        self,other_neuron = self._handle_back_add()
        minus_one = Neuron(-1)
        return self * minus_one
    
    def __sub__(self, other_neuron):
        if not isinstance(other_neuron, Neuron):
            other_neuron = Neuron(other_neuron)
        return self + (-other_neuron)
    
    def __rsub__(self, other_neuron):
        if not isinstance(other_neuron, Neuron):
            other_neuron = Neuron(other_neuron)
        return other_neuron + -(self)
    
    def __truediv__(self,other_neuron):
        if not isinstance(other_neuron, Neuron):
            other_neuron = Neuron(other_neuron)
        
        return self * other_neuron.mul_inverse()
    
    def __rtruediv__(self,other_neuron):
        if not isinstance(other_neuron, Neuron):
            other_neuron = Neuron(other_neuron)
        return self.mul_inverse() * other_neuron
    
    def mul_inverse(self):
        self,other_neuron = self._handle_back_add()
        new_neuron = Neuron(1/self.value)
        self.grad = -1/(self.value**2)
        new_neuron.children = [self]
        return new_neuron
        
    
    def log(self):
        self,other_neuron = self._handle_back_add()
        new_neuron = Neuron(np.log(self.value))
        self.grad = 1/self.value
        new_neuron.children = [self]
        return new_neuron
         
        
    def exp(self):
        self,other_neuron = self._handle_back_add()
        new_neuron = Neuron(np.exp(self.value))
        self.grad = np.exp(self.value)
        new_neuron.children = [self]
        return new_neuron
        
    def backward(self):
        assert self.grad is None
        self.grad = 1
        root = self
        stack = [root]
        while len(stack) != 0:
            root = stack.pop(0)
            for child in root.children:
                if root.is_pointer:
                    child.grad += root.grad
                else:
                    child.grad *= root.grad
                stack.append(child)