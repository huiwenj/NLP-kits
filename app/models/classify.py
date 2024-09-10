import codecs
import os
import random
import string

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
languages = ["af", "cn", "de", "fi", "fr", "in", "ir", "pk", "za"]


def trainOneEpoch(model, criterion, optimizer, X, y):
    """
    Define a function to train the model for one epoch called trainOneEpoch.

    Do the following steps:

    1. Reset any accumulated gradients in the model to zero.
    2. Initialize a hidden state for the model using its initHidden method.
    3. Randomly select a training pair (a category and a line, along with their tensor representations) using the random_training_pair function on X and y.
    4. Loop over each tensor (character) in the line_tensor:
    a. For each tensor, pass it and the current hidden state into the model to get the predicted output and the next hidden state.
    5. Once the entire line_tensor is processed, compute the loss by comparing the model's final output to the true category_tensor using the provided criterion.
    6. Propagate the error backward through the model to compute the gradients.
    7. Update the model's parameters using the optimizer's step method.
    8. Return the model's output, the computed loss as a single value, and the original line and category from the random training pair.

    Inputs:
        - model: the neural network model we want to train
        - criterion: the loss function to calculate the training error
        - optimizer: the optimization algorithm to adjust model parameters
        - X: the input data
        - y: the corresponding labels

    Returns:
        - output: the model's final output (prediction)
        - output_loss: the computed loss as a single value
        - line: the randomly choosen line from random_training_pair()
        - category: the randomly choosen category from random_training_pair()
    """
    # Zeroing the gradients to clear up the accumulated history
    model.zero_grad()
    # Initializing the hidden state for the model
    hidden = model.init_hidden().to(device)

    category, line, category_tensor, line_tensor = random_training_pair(X, y)

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    # Calculating the loss between the model's output and the actual target (category_tensor)
    loss = criterion(output, category_tensor)
    # Backward pass: compute the gradient of the loss with respect to model parameters
    loss.backward()
    # Updating the model parameters based on the calculated gradients
    optimizer.step()
    # Extracting the value of the loss as a Python number
    output_loss = loss.data.item()
    return output, output_loss, line, category


def predict_multi(model, X, y=None, loss_func=None):
    with torch.no_grad():
        model.eval()
        pred = []
        val_loss = []
        for ind in range(X.shape[0]):
            hidden = model.init_hidden().to(device)
            val = line_to_tensor(X[ind])
            for i in range(val.size()[0]):
                output, hidden = model(val[i], hidden)

            # TODO: fill this part to get prediction from output
            # pseudocode:
            #   1. Get the index of the maximum value of the output tensor
            #   2. Append it to the pred list
            _, predicted = torch.max(output, 1)
            pred.append(predicted.item())

            if y is not None and loss_func is not None:
                category_tensor = torch.tensor([int(y[ind])]).to(device)
                val_loss.append(loss_func(output, category_tensor).data.item())

    if y is not None and loss_func is not None:
        return sum(val_loss) / len(val_loss)
    return np.array(pred)


def calculateAccuracy_multi(model, X, y):
    preds = predict_multi(model, X)
    return accuracy_score(preds, y)


def run_multi(
    train_data,
    val_data,
    hidden_size,
    n_epochs,
    learning_rate,
    loss_func,
    print_every,
    plot_every,
    model_name,
):
    X, y = train_data
    X_val, y_val = val_data
    model = RNN_multi(
        input_size=len(all_letters), hidden_size=hidden_size, output_size=len(languages)
    )
    model = model.to(device)
    current_loss = 0
    train_losses = []
    val_losses = []

    for epoch in range(0, n_epochs):
        output, loss, line, category = trainOneEpoch(
            model,
            criterion=loss_func,
            optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate),
            X=X,
            y=y,
        )
        current_loss += loss

        if epoch % print_every == 0:
            # TODO: design your own report to print (freestyle!)
            # What you can do:
            #   1. make prediction
            #   2. compare with gold label to see right or wrong
            #   3. report the number of epoch, the percentage of completion, loss,...
            log_probabilities = output.cpu().data
            prediction = torch.argmax(log_probabilities).item()
            correct = (
                "correct" if prediction == category else f"incorrect (True:{category})"
            )
            print(
                f"Epoch {epoch} ({epoch / n_epochs * 100:.2f}%)  Loss: {loss:.4f}, Word: {line}, Prediction: {prediction} | {correct}"
            )

        if epoch % plot_every == 0:
            train_losses.append(current_loss / plot_every)
            current_loss = 0

            # Validation Loss
            val_loss = predict_multi(model, X_val, y_val, loss_func)
            val_losses.append(val_loss)

    save_dir = "./.models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)

    torch.save(model.state_dict(), model_path)
    return train_losses, val_losses, model_path


class RNN_multi(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        The function should accept various range of output size.

        Inputs:
            self: points to initialized object
            input_size: dimensions of input tensor
            hidden_size: dimensions of the hidden layer
            output_size: dimensions of the expected output tensor
        Returns:
            nothing, it initializes the RNN object
        """
        super(RNN_multi, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()  # Calling the parent class (nn.Module) initializer

        self.hidden_size = hidden_size  # Define the size of the hidden state

        # Linear layer taking concatenated input and hidden state to the next hidden state
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # Linear layer to map hidden state to output
        # A hidden layer in a neural network is between the input and output layers and captures patterns in the data by applying weights and activation functions.
        self.h2o = nn.Linear(hidden_size, 2)
        # LogSoftmax activation for output (useful for classification tasks)
        # The softmax function converts a vector of values into a probability distribution, often used in multi-class classification to assign probabilities to different classes.
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # Concatenate the input and hidden tensors along dimension 1
        combined = torch.cat((input, hidden), 1)
        # Pass the concatenated tensor through the i2h layer to get the next hidden state
        hidden = self.i2h(combined)
        # Pass the hidden state through the h2o layer to get the raw output
        output = self.h2o(hidden)
        # Apply softmax to the raw output
        output = self.softmax(output)
        # Return the final output and the new hidden state
        return output, hidden

    def init_hidden(self):
        # Initializes hidden state with zeros
        return torch.zeros(1, self.hidden_size)


def predict(model, X, y=None, loss_func=None):
    """
    Make predictions on the input data X using the given model.
    Optionally calculate the average loss using true labels y and loss function loss_func.

    Inputs:
        model: trained model
        X: a list of words
        y: a list of categories (optional)
        loss_func: a loss function (optional)
    Returns:
        predictions: as a NumPy array if y and loss_func are None, else the average loss.
    """
    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()
        # Initialize lists to store predictions and individual losses
        pred = []
        val_loss = []
        # Loop over each sample in the input data X
        for ind in range(X.shape[0]):
            # Initialize hidden state
            hidden = model.init_hidden().to(device)
            # Convert the current input sample to a tensor
            val = line_to_tensor(X[ind])
            # Loop over each element in the input tensor and get the model's output
            for i in range(val.size()[0]):
                output, hidden = model(val[i], hidden)
            # Move the output tensor back to CPU and extract data (log probabilities)
            log_probabilities = output.cpu().data
            # Calculate the prediction by comparing the log probabilities
            log_prob0, log_prob1 = log_probabilities[0]
            pred.append(int(log_prob0 < log_prob1))
            # If true labels and a loss function are provided, calculate the loss for the current sample
            if y is not None and loss_func is not None:
                category_tensor = torch.tensor([int(y[ind])]).to(device)
                val_loss.append(loss_func(output, category_tensor).data.item())

    # If true labels and a loss function were provided, return the average loss
    if y is not None and loss_func is not None:
        return sum(val_loss) / len(val_loss)

    # Otherwise, return the predictions as a NumPy array
    return np.array(pred)


def run(
    train_data,
    val_data,
    hidden_size,
    n_epochs,
    learning_rate,
    loss_func,
    print_every,
    plot_every,
    model_name,
):
    X, y = train_data
    X_val, y_val = val_data
    model = RNN(input_size=len(all_letters), hidden_size=hidden_size)
    model = model.to(device)
    current_loss = 0
    train_losses = []
    val_losses = []

    for epoch in range(0, n_epochs):
        output, loss, line, category = train_one_epoch(
            model,
            criterion=loss_func,
            optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate),
            X=X,
            y=y,
        )
        current_loss += loss

        # print intermediate reports
        if epoch % print_every == 0:
            log_probabilities = output.cpu().data
            log_prob0, log_prob1 = log_probabilities[0]
            prediction = int(log_prob0 < log_prob1)
            correct = (
                "correct"
                if prediction == category
                else "incorrect (True:%s)" % category
            )
            print(
                "Epoch %d (%d%%)  Loss: %.4f, Word: %s, Prediction: %s | %s"
                % (epoch, epoch / n_epochs * 100, loss, line, prediction, correct)
            )

        if epoch % plot_every == 0:
            # Training Loss
            train_losses.append(current_loss / plot_every)
            current_loss = 0

            # Validation Loss
            val_losses.append(predict(model, X_val, y_val, loss_func))

    save_dir = "./.models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), model_path)
    return train_losses, val_losses, model_path


def train_one_epoch(model, criterion, optimizer, X, y):
    """
    Define a function to train the model for one epoch called train_one_epoch.

    Do the following steps:

    1. Reset any accumulated gradients in the model to zero.
    2. Initialize a hidden state for the model using its initHidden method.
    3. Randomly select a training pair (a category and a line, along with their tensor representations) using the random_training_pair function on X and y.
    4. Loop over each tensor (character) in the line_tensor:
    a. For each tensor, pass it and the current hidden state into the model to get the predicted output and the next hidden state.
    5. Once the entire line_tensor is processed, compute the loss by comparing the model's final output to the true category_tensor using the provided criterion.
    6. Propagate the error backward through the model to compute the gradients.
    7. Update the model's parameters using the optimizer's step method.
    8. Return the model's output, the computed loss as a single value, and the original line and category from the random training pair.

    Inputs:
        - model: the neural network model we want to train
        - criterion: the loss function to calculate the training error
        - optimizer: the optimization algorithm to adjust model parameters
        - X: the input data
        - y: the corresponding labels

    Returns:
        - output: the model's final output (prediction)
        - output_loss: the computed loss as a single value
        - line: the randomly choosen line from random_training_pair()
        - category: the randomly choosen category from random_training_pair()
    """
    # Zeroing the gradients to clear up the accumulated history
    model.zero_grad()
    # Initializing the hidden state for the model
    hidden = model.init_hidden().to(device)

    category, line, category_tensor, line_tensor = random_training_pair(X, y)

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    # Calculating the loss between the model's output and the actual target (category_tensor)
    loss = criterion(output, category_tensor)
    # Backward pass: compute the gradient of the loss with respect to model parameters
    loss.backward()
    # Updating the model parameters based on the calculated gradients
    optimizer.step()
    # Extracting the value of the loss as a Python number
    output_loss = loss.data.item()
    return output, output_loss, line, category


def random_training_pair(X, y, seed=None):  # seed is required for penngrader only.
    """
    Pseudocode:
        1. Initialize a random generator with given seed.
        2. Generate a random index 'ind' between 0 and (number of rows in X) - 1.
        3. Fetch 'category' from y and 'line' from X using the random index 'ind'.
        4. Convert 'category' to a tensor and move it to the specified device.
        5. Convert 'line' to a tensor by calling the function line_to_tensor.
        6. Return 'category', 'line', 'category_tensor', and 'line_tensor'.

    Input:
        training data:
            X: features
            y: labels
            seed: needed for randomness

    Returns:
        A tuple of 4 items:
            category: output label(category) as an integer,
            line: input line (here by word) as a string,
            category_tensor: the category as a tensor. Ex) category = 1 => category_tensor = tensor([1]),
                            Tip: make sure to send your tensor to GPU!
            line_tensor: line as a tensor. Tip: use line_to_tensor()!
    """
    if seed is not None:
        random.seed(seed)

    ind = random.randint(0, len(X) - 1)
    category = y[ind]
    line = X[ind]

    category_tensor = torch.tensor([category], dtype=torch.long).to(device)
    line_tensor = line_to_tensor(line)

    return category, line, category_tensor, line_tensor


# Turn a line into a <line_length x 1 x n_letters>
# input: a line of text
# output:  a <line_length x 1 x n_letters> tensor
def line_to_tensor(line):
    """
    Turn a line into a <line_length x 1 x n_letters>
    input: a line of text
    output:  a <line_length x 1 x n_letters> tensor

    Inputs:
        line: string signifying a line of text

    Returns:
        a tensor
    """
    line_length = len(line)
    tensor = torch.zeros(line_length, 1, n_letters)

    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor.to(device)


def letter_to_index(letter):
    """
    Find letter index from all_letters, e.g. "a" = 0
    hint: use .find() function
        This could be a one line function!

    Inputs:
        letter: a character. Ex) 'a', 'r', 'T'

    Returns:
        index integer. Ex) 0, 17,
    """
    return all_letters.find(letter)


def load_labeled_file(data_file):
    words = []
    labels = []
    with open(data_file, "rt", encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    X = np.array(words)
    y = np.array(labels)
    return X, y


def getWords(baseDir, lang, train=True):
    suff = "train/" if train else "val/"
    arr = []
    with codecs.open(
        baseDir + suff + lang + ".txt", "r", encoding="utf-8", errors="ignore"
    ) as fp:
        for line in fp:
            arr.append(line.rstrip("\n"))
    return np.array(arr)


def read_data(baseDir, train=True):
    X, y = np.array([]), np.array([])
    for lang in languages:
        tempX = getWords(baseDir, lang, train)
        X = np.append(X, tempX)
        y = np.append(y, np.array([lang] * tempX.shape[0]))
    return X, y


def calculateAccuracy(model, X, y):
    """
    HINT: you can use accuracy_score function.

    Pseudocode:
    1. Calculate prediction of
    X using predict()
    2. Calculate accuracy using accuracy_score() fuction

    Inputs:
        model: trained model,
        X: a list of words,
        y: a list of class labels as integers
    Returns:
        accuracy score of the given model on the given input X and target y
    """
    predictions = predict(model, X)
    return accuracy_score(y, predictions)


def replace_nan_with_none(data):
    if isinstance(data, dict):
        return {k: replace_nan_with_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_nan_with_none(i) for i in data]
    elif isinstance(data, float) and np.isnan(data):
        return None
    else:
        return data