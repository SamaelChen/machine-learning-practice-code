import mxnet as mx
from mxnet import nd, autograd
import matplotlib.pyplot as plt

mx.random.seed(1)

num_inputs = 2
num_outputs = 1
num_examples = 10000

X = nd.random_normal(shape=(num_examples, num_inputs))
y = 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2 + 0.1 * \
    nd.random_normal(shape=(num_examples,))

print(X[0])

batch_size = 4
train_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(X, y),
                                      batch_size=batch_size, shuffle=True)

w = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=num_outputs)
params = [w, b]

for param in params:
    param.attach_grad()


def net(X):
    """Net"""
    return mx.nd.dot(X, w) + b


def square_loss(yhat, y):
    """Loss function"""
    return nd.mean((yhat - y) ** 2)


def SGD(params, lr):
    """SGD"""
    for param in params:
        param[:] = param - lr * param.grad


epochs = 2
ctx = mx.cpu()
learning_rate = 0.001
smoothing_constant = 0.01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx).reshape((-1, 1))
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else(1 - smoothing_constant) * moving_loss +
                       (smoothing_constant) * curr_loss)
        if (i + 1) % 500 == 0:
            print("Epoch %s, batch %s. Moving avg of loos: %s" %
                  (e, i, moving_loss))

# %%
pred = mx.nd.dot(X, params[0]) + params[1]
# %%
plt.scatter(pred.asnumpy(), y.asnumpy())
plt.xlabel('predict')
plt.ylabel('real')
plt.show()
