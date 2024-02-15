import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def function_mapper(self, function_name):
        if function_name == 'relu':
            return torch.relu
        elif function_name == 'sigmoid':
            return torch.sigmoid
        elif function_name == 'identity':         
            return lambda x: x
        else:
            raise ValueError(f'function_name {function_name} not supported')
    
    def function_derivative(self, function_name):
        if function_name == 'relu':
            def relu_derivative(x):
                return (x > 0).float()
            return relu_derivative
        elif function_name == 'sigmoid':
            def sigmoid_derivative(x):
                return torch.sigmoid(x) * (1 - torch.sigmoid(x))
            return sigmoid_derivative
        elif function_name == 'identity':
            def identity_derivative(x):
                return torch.ones_like(x)
            return identity_derivative
        else:
            raise ValueError(f'function_name {function_name} not supported')

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        x = x.t()

        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
    
        s1 = W1 @ x + b1[:, None]

        a1 = self.function_mapper(self.f_function)(s1)

        s2 = W2 @ a1 + b2[:, None]
        y_hat = self.function_mapper(self.g_function)(s2)

        self.cache['x'] = x
        self.cache['s1'] = s1
        self.cache['a1'] = a1
        self.cache['s2'] = s2
        self.cache['y_hat'] = y_hat

        return y_hat.t()
        
        pass
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        dJdy_hat = dJdy_hat.t()
        dJds2 = dJdy_hat * self.function_derivative(self.g_function)(self.cache['s2'])


        self.grads['dJdb2'] = dJds2.sum(1)
        self.grads['dJdW2'] = dJds2 @ self.cache['a1'].t()

        dJda1 = dJds2.t() @ self.parameters['W2']
        dJds1 = dJda1.t() * self.function_derivative(self.f_function)(self.cache['s1'])

        self.grads['dJdb1'] = dJds1.sum(1)
        self.grads['dJdW1'] = dJds1 @ self.cache['x'].t()

        pass

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    loss = ((y - y_hat) ** 2).mean()
    dJdy_hat = -2 * (y - y_hat) / (y.shape[0]*y.shape[1])
    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    loss = - (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat)).mean()
    dJdy_hat = - (y / y_hat - (1 - y) / (1 - y_hat)) / (y.shape[0]*y.shape[1])
    return loss, dJdy_hat











