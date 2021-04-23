import numpy as np
import time



class NN(object):
    def __init__(self,
                 hidden_dims=(256,120),
                 n_classes=2,
                 epsilon=1e-6,
                 lr=0.03,
                 batch_size=5,
                 seed=None,
                 activation="relu",
                 init_method="glorot"):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon


        self.train_logs = {'train_accuracy': [], 'train_loss': []}

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            n_in,n_out=all_dims[layer_n-1], all_dims[layer_n]
            a=np.sqrt(6/(n_in+n_out))
            lower,upper= -a,a

            self.weights[f"W{layer_n}"]=np.random.uniform(-a,a,[all_dims[layer_n-1],all_dims[layer_n]])
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def relu(self, x, grad=False):
        x = np.array(x,dtype=np.float64)
        if grad:
            # WRITE CODE HERE
            relu_x = np.ones_like(x,dtype=np.float64)
            relu_x[x <= 0] = 0
            return relu_x
        else:
            # WRITE CODE HERE
            return np.where(x<=0,0,x)

    def sigmoid(self, x, grad=False):
        if grad:
          # WRITE CODE HERE
            s = 1 / (1 + np.exp(-np.array(x)))
            s_out = s*(1-s)
        else:
            # WRITE CODE HERE
            s_out = 1 / (1 + np.exp(-np.array(x)))
        return s_out

    def tanh(self, x, grad=False):
        if grad:
            # WRITE CODE HERE
            t = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
            t_out = 1 - t**2
        else:
            # WRITE CODE HERE
            t_out = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return t_out

    def leakyrelu(self, x, grad=False):
        alpha = 0.01
        if grad:
            # WRITE CODE HERE
            return np.clip(x >= 0, alpha, 1.0)
        else:
            # WRITE CODE HERE
            le_out = np.copy(x)
            le_out[le_out < 0] *= alpha
            return le_out

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            # WRITE CODE HERE
            out=self.relu(x,grad)
        elif self.activation_str == "sigmoid":
            # WRITE CODE HERE
            out=self.sigmoid(x,grad)
        elif self.activation_str == "tanh":
            # WRITE CODE HERE
            out=self.tanh(x,grad)
        elif self.activation_str == "leakyrelu":
            # WRITE CODE HERE
            out=self.leakyrelu(x,grad)
        else:
            raise Exception("invalid")
        return out

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        # WRITE CODE HERE
        ex = np.exp(x - np.max(x,axis=-1,keepdims=True))
        return  ex/ex.sum(axis=-1,keepdims=True)

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        # WRITE CODE HERE
        for layer_n in range(1,self.n_hidden + 2):
            cache[f"A{layer_n}"]=  np.dot(cache[f"Z{layer_n-1}"],self.weights[f"W{layer_n}"]) + self.weights[f"b{layer_n}"]
            if layer_n==self.n_hidden + 1:
                cache[f"Z{layer_n}"]=self.softmax(cache[f"A{self.n_hidden + 1}"])
            else:
                cache[f"Z{layer_n}"]=  self.activation(cache[f"A{layer_n}"],grad=False)
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        # WRITE CODE HERE
        deri={}
        for i in range(1,self.n_hidden+2):
            deri[f"dAct{i}"]=self.activation(cache[f"A{i}"],grad=True)
        self.deri=deri
        grads[f"dA{self.n_hidden+1}"]=(output-labels)

        for layer_n in reversed(range(1,self.n_hidden + 2)):
            if layer_n==1:
                grads[f"dW{layer_n}"]= np.dot(cache[f"Z{layer_n-1}"].T,grads[f"dA{layer_n}"])/self.batch_size
                grads[f"db{layer_n}"]= np.mean(grads[f"dA{layer_n}"],axis=0,keepdims=True)
            else:
                grads[f"dW{layer_n}"]= np.dot(cache[f"Z{layer_n-1}"].T,grads[f"dA{layer_n}"])/self.batch_size
                grads[f"db{layer_n}"]= np.mean(grads[f"dA{layer_n}"],axis=0,keepdims=True)
                grads[f"dZ{layer_n-1}"]= np.dot(grads[f"dA{layer_n}"],self.weights[f"W{layer_n}"].T)
                grads[f"dA{layer_n-1}"]= np.multiply(grads[f"dZ{layer_n-1}"],self.deri[f"dAct{layer_n-1}"])
        return grads

    def update(self, grads):
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            self.weights[f"W{layer_n}"]= self.weights[f"W{layer_n}"]- self.lr * grads[f"dW{layer_n}"]
            self.weights[f"b{layer_n}"]= self.weights[f"b{layer_n}"]- self.lr * grads[f"db{layer_n}"]

    def one_hot(self, y):
        # WRITE CODE HERE
        classes=np.unique(y)
        one_hot_y= np.zeros([len(y),self.n_classes])
        for i in range(len(classes)):
            one_hot_y[:,i]=np.where(y==classes[i],1,one_hot_y[:,i])
        return one_hot_y

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        # WRITE CODE HERE
        cost=-(np.sum(np.mean(labels*np.log(prediction),axis=0),axis=0,keepdims=True))
        return cost

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self,X_train,y_train, n_epochs):
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):

                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                # WRITE CODE HERE
                self.cache = self.forward(minibatchX)
                self.grads=self.backward(self.cache, minibatchY)
                self.update(self.grads)

            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['train_loss'].append(train_loss)



    def evaluate(self,X_test,y_test):
        test_loss, test_accuracy, prediction = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss,test_accuracy, prediction


if __name__ == "__main__":
    # WRITE CODE HERE
    # Instantiate, train, and evaluate your classifiers in the space below
    pass
