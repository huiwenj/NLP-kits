import torch
from torch import nn

from app.models.classify import all_letters, RNN, device, train_one_epoch, load_labeled_file, run, calculateAccuracy


class ClassifyService:
    def __init__(self):
        pass

    def classify(self, image):
        pass

    def train(self):
        word_train_data = load_labeled_file("data/complex_words_training.txt")
        word_val_data = load_labeled_file("data/complex_words_development.txt")

        word_train_losses, word_val_losses = run(train_data=word_train_data,
                                                 val_data=word_val_data,
                                                 hidden_size=50,
                                                 n_epochs=50000,
                                                 learning_rate=0.005,
                                                 loss_func=nn.NLLLoss(),
                                                 print_every=5000,
                                                 plot_every=250,
                                                 model_name="./word_RNN"
                                                 )

        test_model = RNN(input_size=len(all_letters), hidden_size=50).to(device)
        test_model.load_state_dict(torch.load("word_RNN"))
        val_acc = calculateAccuracy(test_model, word_val_data[0], word_val_data[1])

        return {
            "train_losses": word_train_losses,
            "val_losses": word_val_losses,
        }


