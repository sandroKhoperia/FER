import configuration
import torch
import emotion_cnn
from torch.optim import Adam
from sklearn.metrics import confusion_matrix

def main(use_pretrained):
    data_model = configuration.DataModel()
    model = emotion_cnn.EmotionCNN()
    model.to(data_model.device)
    emotion_cnn.save_model_diagram(model, data_model)
    if use_pretrained:
        model_state_dict = torch.load('../venv/results/model_new.pth')
        model.load_state_dict(model_state_dict)
        optimizer = Adam(model.parameters(), lr= data_model.LEARNING_RATE)
        optimizer_state_dict = torch.load('../venv/results/optimizer_new.pth')
        optimizer.load_state_dict(optimizer_state_dict)

    else:
        emotion_cnn.train_process(model, data_model, save_model=True)
    emotion_cnn.plot_confussion_matrix(model, data_model)
    emotion_cnn.output_predictions(model, data_model, data_model)

if __name__ == '__main__':
    main(True)