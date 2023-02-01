#Gradient Calculation using Computation Graph


TODO:
- [ ] Support addition of gradients during backward prop  
When a Neuron is used in computation twice its gradients should be added during backward prop.  
for example:  
$y = a \times a$  
Right now the gradient of a would be the value of $a$ and not $2a$.  
- [ ] Support matrix operations
- [ ] Train a NN on MNIST 

