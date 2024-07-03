# LLM Project

## Project Task
This project involves performing sentiment analysis on movie reviews using a pre-trained language model. The goal is to classify each review as either positive or negative. The project includes fine-tuning a pre-trained model, evaluating its performance, optimizing it through hyperparameter tuning, and using it for inference on new, unseen data.

## Dataset
The dataset used for this project is the IMDb dataset, which is a collection of 50,000 movie reviews labeled as either positive or negative. It is available through the Hugging Face datasets library:

```python
from datasets import load_dataset

data = load_dataset('imdb')
```

## Pre-trained Model
The pre-trained model selected for this project is **`distilbert-base-uncased-finetuned-sst-2-english`**. This model is a distilled version of BERT (Bidirectional Encoder Representations from Transformers) fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset, which is specifically designed for sentiment analysis.


## Performance Metrics
The performance of the model is evaluated using **`accuracy`** metric, which is the proportion of correctly classified reviews out of the total number of reviews.

Results after fine-tuning the model:

- **Accuracy: 97%**
<p align="center" style="margin-top: 20px; margin-bottom: 20px;">
<img width="60%" src="https://github.com/ThuyTran102/LLM-Project/blob/main/images/result.png" alt="Outcome"></img>
</p>

>> The pre-trained LLM without fine-tuning performs slightly better than classical models, showing the inherent power of such models in text understanding. However, the fine-tuned pre-trained LLM significantly outperforms all other models, reaching an accuracy of 97%. This highlights the importance of fine-tuning for domain-specific tasks, leveraging the extensive pre-training while adapting to the specific characteristics of the target dataset.

## Key Hyperparameters and their importance
When fine-tuning a pre-trained language model, selecting the right hyperparameters is crucial for achieving optimal performance. The following hyperparameters were found to be most important and relevant during the optimization of the model:

1. **Learning Rate** (`learning_rate`):

-Value: 2e-5
- Importance: The learning rate controls how much the model weights are adjusted with respect to the loss gradient. A well-chosen learning rate ensures that the model converges efficiently without overshooting the optimal parameters. In this case, **2e-5** was found to be a good balance for stable and effective learning.

2. **Batch Size** (`per_device_train_batch_size`, `per_device_eval_batch_size`):

- Value: 16
- Importance: The batch size determines the number of samples processed before the model's internal parameters are updated. A batch size of 16 was chosen to balance between computational efficiency and stability of the gradient estimates. Smaller batch sizes can lead to more stable but slower convergence, while larger sizes can speed up training but may require more memory and could lead to less stable gradients.

3. **Number of Epochs** (`num_train_epochs`):

- Value: 3
- Importance: The number of epochs indicates how many times the entire training dataset passes through the model. Setting this to 3 epochs allowed sufficient iterations for the model to learn the patterns in the dataset without overfitting.

4. **Weight Decay** (`weight_decay`):

- Value: 0.01
- Importance: Weight decay is a regularization technique to prevent overfitting by penalizing large weights. A weight decay of 0.01 was effective in maintaining generalization while avoiding overfitting.

5. **Warmup Steps** (`warmup_steps`):

- Value: 100
- Importance: Warmup steps gradually increase the learning rate from 0 to the initial learning rate during the first few steps of training. This helps in stabilizing the training process. Setting it to 100 steps helped to avoid large updates early in training that could destabilize learning.

6. **Evaluation Strategy** (`evaluation_strategy`):

- Value: "epoch"
- Importance: This parameter controls how often the model is evaluated on the validation set. Evaluating at the end of each epoch helps in monitoring the model's performance and making early adjustments if necessary.

7. **Save Strategy** (`save_strategy`):

- Value: "epoch"
- Importance: Saving the model at the end of each epoch ensures that checkpoints are created regularly, allowing recovery and further tuning from the best-performing models.

8. **Logging Steps** (`logging_steps`):

- Value: 100
- Importance: This determines how frequently training progress is logged. Regular logging (every 100 steps) provides a detailed view of the training process, helping to diagnose issues and track performance improvements.

9. **Load Best Model at End** (`load_best_model_at_end`):

- Value: True
- Importance: Automatically loading the best model at the end of training ensures that the final model used is the one that performed best on the validation set, thus providing the optimal results.

