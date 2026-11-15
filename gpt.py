import numpy as np 

class Embedding_layer:
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random,randn(voacb_size, embedding_dim) * 0.02
        self.inputs = None

        self.grad_embeddings = np.zeros(self.embeddings.shape)

    def forward(self, input):
        self.inputs = input
        return self.embeddings[input]
    def backward(self, prev_grad):
        grad_embeddings = np.zeros(self.embeddings.shape)
        np.add.at(grad_embeddings, self.inputs, prev_grad)

        self.grad_embeddings += grad_embeddings
        return grad_embeddings

#converts raw final ouputs (logit) into probabilities
class softmax:
    def __init__(self):
        self.softmax_out = None

    def forward(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        self.softmax_out = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        softmax_out = self.softmax_out
        return softmax_out
class Linear:
    def __init__(self, dim_in, dim_out, include_bias=True):
        scale = np.sqrt(1.0 / dim_in)
        self.W = np.random.randn(dim_in, dim_out) * scale
        self.b = np.zeros(dim_out) if include_bias else None
    def __call__(self,x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        out = x @ self.w
        if self.b is not None:
            out += self.b
        return out
    def backward(self, grad_out):
        x_flat = self.x.reshape(-1, self.x.shape[-1])
        grad_flat = grad_out.reshape(-1, grad_out.shape[-1])
        self.dW = x_flat.T @ grad_flat
        self.db = np.sum(grad_flat, axis=0) if self.b is not None else None
        grad_in = grad_out @ self.W.T
        return grad_in 
    

class Multihead_attention:
    def __init__(self, dim_in, dim_out, n_heads):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_heads = n_heads

        self.query_linear = Linear(dim_in, dim_out, include_bias=False)
        self.key_linear = Linear(dim_in, dim_out, include_bias=False)
        self.value_linear = Linear(dim_in, dim_out, include_bias=False)
        self.softmax = softmax()
        self.final_linear = Linear(dim_out, dim_out, include_bias=False)

        self.queries_matrix = None
        self.keys_matrix = None
        self.values_matrix = None
        self.attention_weights = None
    def forward(self, inputs, mask=None):
        B, T, D = inputs.shape

        queries = self.query_linear.forward(inputs)
        keys = self.key_linear.forward(inputs)
        values = self.value_linear.forward(inputs)

        queries = np.reshape(queries, (B, T, self.n_heads, self.dim_out // self.n_heads))
        keys = np.reshape(keys, (B, T, self.n_heads, self.dim_out // self.n_heads))
        values = np.reshape(values, (B, T, self.n_heads, self.dim_out // self.n_heads))

        queries = np.transpose(queries, (0, 2, 1, 3))
        keys = np.transpose(keys, (0, 2, 1, 3))
        values = np.transpose(values, (0, 2, 1, 3))

        attention_scores = queries @ np.transpose(keys, (0,1,3,2))
        attention_scores = attention_scores / np.sqrt(keys.shape[-1])
        if mask is not None:
            attention_scores+= mask
        attention_weights = self.softmax.forward(attention_scores)
        final_vector = attention_weights @ values
        final_vector = np.transpose(final_vector, (0, 2, 1, 3))
        final_vector = np.reshape(final_vector, (B, T, self.dim_out))
        final_output = self.final_linear.forward(final_vector)

        self.queries_matrix = queries
        self.keys_matrix = keys
        self.values_matrix = values
        self.attention_weights = attention_weights

        return final_output 
        
        