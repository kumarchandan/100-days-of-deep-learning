
def update_w_and_b(spendings, sales, w, b, alpha):
    dl_dw = 0.0
    dl_db = 0.0
    N = len(spendings)
    # df/dw = -2x_i (y_i - (wx_i + b))
    # df/db = -2 (y_i - (wx_i + b))

    # Collect gradients (via partial derivatives) on entire dataset to find the best values for parameters
    for i in range(N):
        dl_dw += -2 * spendings[i] * (sales[i] - (w * spendings[i] + b))
        dl_db += -2 * (sales[i] - (w * spendings[i] + b))
    
    # Update weights of parameters w and b
    w = w - (1 / float(N)) * dl_dw * alpha
    b = b -  (1 / float(N)) * dl_db * alpha
    return w, b

def avg_loss(spendings, sales, w, b):
    N = len(spendings)
    total_error = 0.0

    # Calculate MSE
    for i in range(N):
        total_error += (sales[i] - (w * spendings[i] + b)) ** 2
    total_error = total_error / float(N)
    return total_error

def train(spendings, sales, w, b, alpha, epochs):
    # values of w and b are initialized with 0
    
    # each epoch takes entire dataset
    for e in range(epochs):
        w, b = update_w_and_b(spendings, sales, w, b, alpha)

        # Log the progress (Show only for first 50 epochs)
        if e // 50 == 0:
            print('epoch: ', e, ' loss: ', avg_loss(spendings, sales, w, b), ' w and b: ', w, b)
    return w, b

def predict(x, w, b):
    return w * x + b
