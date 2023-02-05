# Gradient Calculation using Computation Graph


TODO:
- [x] Support addition of gradients during backward prop  
- [x] Support reduce functions like sum, max, min etc.
- [x] Support matrix operations
- [ ] Test every form of gradient using pytorch
- [ ] Fix addition reduction bug in backpass
- [ ] Train a NN on MNIST 

Notes:
The numpy broadcasting system is very confusion.  
First don't allow element-wise operations without same shapes
Later on can write a broadcaster

Also can we write a C matmul and make a binding. It would be interesting to write a fast matmul.

May also be useful to visualize the model, forward and backward to debug.
