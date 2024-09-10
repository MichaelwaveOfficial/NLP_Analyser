from sklearn.utils import compute_class_weight
import numpy as np 
import evaluate


metrics = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metrics.compute(predictions=predictions, references=labels)


def get_class_weights(dataframe):
    return compute_class_weight('balanced', classes = sorted(dataframe['label'].unique().tolist()), y = dataframe['label'].tolist())
