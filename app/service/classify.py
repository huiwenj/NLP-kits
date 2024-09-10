import os

import torch
from torch import nn

from app.models.classify import all_letters, RNN, device, load_labeled_file, run, calculateAccuracy, line_to_tensor
from app.models.response import R


# This is a service class that contains the logic for training and predicting the model
class ClassifyService:
    def train(self):
        word_train_data = load_labeled_file("data/complex_words_training.txt")
        word_val_data = load_labeled_file("data/complex_words_development.txt")

        word_train_losses, word_val_losses, model_path = run(train_data=word_train_data,
                                                             val_data=word_val_data,
                                                             hidden_size=50,
                                                             n_epochs=10000,
                                                             learning_rate=0.005,
                                                             loss_func=nn.NLLLoss(),
                                                             print_every=5000,
                                                             plot_every=250,
                                                             model_name="word_RNN"
                                                             )

        test_model = RNN(input_size=len(all_letters), hidden_size=50).to(device)
        test_model.load_state_dict(torch.load(model_path))
        val_acc = calculateAccuracy(test_model, word_val_data[0], word_val_data[1])

        return {
            "train_losses": word_train_losses,
            "val_losses": word_val_losses,
            "val_acc": val_acc
        }

    def predict(self, word):
        """
        Predict the word is simple or hard
        Args:
            word: str

        Returns: 1 is complex, 0 is simple

        """
        if not os.path.exists("./.models/word_RNN"):
            return R.bad_request("Model not found, please train the model first")

        try:
            model = RNN(input_size=len(all_letters), hidden_size=50).to(device)
            model.load_state_dict(torch.load("./.models/word_RNN"))
        except Exception as e:
            return R.error(f"Error when loading model: {str(e)}")

        try:
            with torch.no_grad():
                model.eval()
                line_tensor = line_to_tensor(word)
                hidden = model.init_hidden().to(device)

                for i in range(line_tensor.size()[0]):
                    output, hidden = model(line_tensor[i], hidden)

                topv, topi = output.topk(1)
                category_index = topi[0].item()
                return R.success(category_index)
        except Exception as e:
            return R.error(f"Error when predicting: {str(e)}")
