# https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments
# https://discuss.pytorch.org/t/runtimeerror-trying-to-backward-through-the-graph-a-second-time-but-the-buffers-have-already-been-freed-specify-retain-graph-true-when-calling-backward-the-first-time/6795
# 解释pytorch的example中关于backward()和gradients的内容
from torch.autograd import Variable
import torch

x = Variable(torch.FloatTensor([[1, 2, 3, 4]]), requires_grad=True)
print(x)
# z = 2 * x
z = x * x
print(z)
loss = z.sum(dim=1)
print(loss)

# do backward for first element of z
z.backward(torch.FloatTensor([[1, 0, 0, 0]]), retain_graph=True)
print(x.grad.data)
x.grad.data.zero_()  # remove gradient in x.grad, or it will be accumulated

# do backward for second element of z
z.backward(torch.FloatTensor([[0, 1, 0, 0]]), retain_graph=True)
print(x.grad.data)
x.grad.data.zero_()

# do backward for second element of z
z.backward(torch.FloatTensor([[0, 2, 0, 0]]), retain_graph=True)
print(x.grad.data)
x.grad.data.zero_()

# do backward for second element of z
z.backward(torch.FloatTensor([[0.2, 1, 0, 0]]), retain_graph=True)
print(x.grad.data)
x.grad.data.zero_()

# do backward for second element of z
z.backward(torch.FloatTensor([[0.1, 1, 0, 0.0001]]), retain_graph=True)
print(x.grad.data)
x.grad.data.zero_()

# do backward for all elements of z, with weight equal to the derivative of
# loss w.r.t z_1, z_2, z_3 and z_4
z.backward(torch.FloatTensor([[1, 1, 1, 1]]), retain_graph=True)
print(x.grad.data)
x.grad.data.zero_()

# do backward for all elements of z, with weight equal to the derivative of
# loss w.r.t z_1, z_2, z_3 and z_4
z.backward(torch.FloatTensor([[2, 2, 2, 2]]), retain_graph=True)
print(x.grad.data)
x.grad.data.zero_()

# or we can directly backprop using loss
loss.backward()  # equivalent to loss.backward(torch.FloatTensor([1.0]))
print(x.grad.data)


print("----------------------")
a = Variable(torch.FloatTensor([3]), requires_grad=True)
b = a * a
print(a)
print(b)
grad = torch.tensor([5], dtype=torch.float)
b.backward(grad)
print(a.grad.data)
