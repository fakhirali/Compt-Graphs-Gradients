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
