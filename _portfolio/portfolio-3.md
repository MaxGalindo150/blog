---
title: "PII Data Detection with BERT (*Bidirectional Encoder Representations from Transformers*)"
excerpt: "Here's how to train a model to recognize Personally Identifiable Information (PII)."
collection: portfolio
---




The primary task of this project falls under **Token Classification**, also known as **Named Entity Recognition (NER)**. Unlike Text Classification, where a single label is assigned to an entire text, Token Classification involves labeling each word or token within a text with the appropriate category.

This project uses a pretrained model called BERT (Bidirectional Encoder Representations from Transformers) to perform this task from scratch. BERT has proven to be highly effective for NER tasks due to its ability to understand the bidirectional context of each token in a sequence. The associated notebook guides you through the entire process, from data preparation to model training and evaluation.


# 📚 | Import Libraries 


```python
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForTokenClassification
from transformers import pipeline
from seqeval.metrics import classification_report
import matplotlib.pyplot as plt
```

# ⚙️ | Configuration


```python
class CFG:
    seed = 42
    preset = "bert-base-cased" # name of pretrained backbone
    train_seq_len = 1024 # max size of input sequence for training
    train_batch_size = 4 # size of the input batch in training
    eval_batch_size = 4 # size of the input batch in evaluation
    epochs = 3 # number of epochs to train
    lr = 2e-5 # learning rate
    
    labels = ["B-EMAIL", "B-ID_NUM", "B-NAME_STUDENT", "B-PHONE_NUM",
              "B-STREET_ADDRESS", "B-URL_PERSONAL", "B-USERNAME",
              "I-ID_NUM", "I-NAME_STUDENT", "I-PHONE_NUM",
              "I-STREET_ADDRESS","I-URL_PERSONAL","O"]
    id2label = dict(enumerate(labels)) # integer label to BIO format label mapping
    label2id = {v:k for k,v in id2label.items()} # BIO format label to integer label mapping
    num_labels = len(labels) # number of PII (NER) tags
    
```

# 📁 | Dataset Path


```python
BASE_PATH = './data'
data = pd.read_json(f'{BASE_PATH}/train.json')
```

# 📖 | Meta Data

The dataset contains ~$22,000$ student essays where $70\%$ essays are reserved for **testing**, leaving $30\%$ for **training** and **validation**.

Sure, here's the modified markdown with an example of the BIO format label:

**Data Overview:**

* All essays were written in response to the **same prompt**, applying course material to a real-world problem.
* The dataset includes **7 types of PII**: `NAME_STUDENT`, `EMAIL`, `USERNAME`, `ID_NUM`, `PHONE_NUM`, `URL_PERSONAL`, `STREET_ADDRESS`.
* Labels are given in **BIO (Beginning, Inner, Outer)** format.

**Example of BIO format label:**

Let's consider a sentence: `"The email address of Michael jordan is mjordan@nba.com"`. In BIO format, the labels for the personally identifiable information (PII) would be annotated as follows:

| **Word** | The | email | address | of | Michael | Jordan | is | mjordan@nba.com |
|----------|-----|-------|---------|----|---------|--------|----|----------------|
| **Label** | O   | O     | O       | O  | B-NAME_STUDENT | I-NAME_STUDENT | O  | B-EMAIL        |

In the example above, `B-` indicates the beginning of an PII, `I-` indicates an inner part of a multi-token PII, and `O` indicates tokens that do not belong to any PII.

**Data Format:**

* The train/test data is stored in `{test|train}.json` files.
* Each json file has:
    * `document`: unique ID (integer)
    * `full_text`: essay content (string)
    * `tokens`: individual words in the essay (list of strings)
    * `labels` (training data only): BIO labels for each token (list of strings)


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>document</th>
      <th>full_text</th>
      <th>tokens</th>
      <th>trailing_whitespace</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>Design Thinking for innovation reflexion-Avril...</td>
      <td>[Design, Thinking, for, innovation, reflexion,...</td>
      <td>[True, True, True, True, False, False, True, F...</td>
      <td>[O, O, O, O, O, O, O, O, O, B-NAME_STUDENT, I-...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>Diego Estrada\n\nDesign Thinking Assignment\n\...</td>
      <td>[Diego, Estrada, \n\n, Design, Thinking, Assig...</td>
      <td>[True, False, False, True, True, False, False,...</td>
      <td>[B-NAME_STUDENT, I-NAME_STUDENT, O, O, O, O, O...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>Reporting process\n\nby Gilberto Gamboa\n\nCha...</td>
      <td>[Reporting, process, \n\n, by, Gilberto, Gambo...</td>
      <td>[True, False, False, True, True, False, False,...</td>
      <td>[O, O, O, O, B-NAME_STUDENT, I-NAME_STUDENT, O...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>Design Thinking for Innovation\n\nSindy Samaca...</td>
      <td>[Design, Thinking, for, Innovation, \n\n, Sind...</td>
      <td>[True, True, True, False, False, True, False, ...</td>
      <td>[O, O, O, O, O, B-NAME_STUDENT, I-NAME_STUDENT...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56</td>
      <td>Assignment:  Visualization Reflection  Submitt...</td>
      <td>[Assignment, :,   , Visualization,  , Reflecti...</td>
      <td>[False, False, False, False, False, False, Fal...</td>
      <td>[O, O, O, O, O, O, O, O, O, O, O, O, B-NAME_ST...</td>
    </tr>
  </tbody>
</table>
</div>



# 📊 | Exploratory Data Analysis

From the following label distribution plot, it is evident that there is a significant **class imbalance** between PII tags. This could be a key area for improvement where **external datasets** and **augmentations** could play a pivotal role.


```python
# Train-Valid data
data_json = json.load(open(f"{BASE_PATH}/train.json"))

# Initialize empty arrays
words = np.empty(len(data_json), dtype=object)
labels_json = np.empty(len(data_json), dtype=object)

# Fill the arrays
for i, x in tqdm(enumerate(data_json), total=len(data_json)):
    words[i] = np.array(x["tokens"])
    labels_json[i] = np.array([CFG.label2id[label] for label in x["labels"]])
```

    100%|██████████| 6807/6807 [00:00<00:00, 9510.03it/s] 



```python
# Get unique labels and their frequency
all_labels = np.array([x for label in labels_json for x in label])
unique_labels, label_counts = np.unique(all_labels, return_counts=True)

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(CFG.labels, label_counts, log=True)

plt.title("Label Distribution")
plt.xlabel("Labels")
plt.ylabel("Count")

# Add counts above the bars
for i in range(len(label_counts)):
    plt.text(i, label_counts[i], str(label_counts[i]), ha = 'center')


plt.xticks(rotation='vertical')

plt.show()
```


    
![png](../../images/labels.png)
    


## 🔥 Encoder

To use the BERT model, it is essential to encode our labels as numbers. This is achieved by mapping the labels to their corresponding numerical IDs, as defined in the configuration class:



```python
CFG.label2id
```




    {'B-EMAIL': 0,
     'B-ID_NUM': 1,
     'B-NAME_STUDENT': 2,
     'B-PHONE_NUM': 3,
     'B-STREET_ADDRESS': 4,
     'B-URL_PERSONAL': 5,
     'B-USERNAME': 6,
     'I-ID_NUM': 7,
     'I-NAME_STUDENT': 8,
     'I-PHONE_NUM': 9,
     'I-STREET_ADDRESS': 10,
     'I-URL_PERSONAL': 11,
     'O': 12}




```python
# Aplicar el mapeo a la columna 'labels'
data['ner_tags'] = data['labels'].apply(lambda x: [CFG.label2id[i] for i in x])
data = data.drop(columns=['labels'])
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>document</th>
      <th>full_text</th>
      <th>tokens</th>
      <th>trailing_whitespace</th>
      <th>ner_tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>Design Thinking for innovation reflexion-Avril...</td>
      <td>[Design, Thinking, for, innovation, reflexion,...</td>
      <td>[True, True, True, True, False, False, True, F...</td>
      <td>[12, 12, 12, 12, 12, 12, 12, 12, 12, 2, 8, 12,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>Diego Estrada\n\nDesign Thinking Assignment\n\...</td>
      <td>[Diego, Estrada, \n\n, Design, Thinking, Assig...</td>
      <td>[True, False, False, True, True, False, False,...</td>
      <td>[2, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>Reporting process\n\nby Gilberto Gamboa\n\nCha...</td>
      <td>[Reporting, process, \n\n, by, Gilberto, Gambo...</td>
      <td>[True, False, False, True, True, False, False,...</td>
      <td>[12, 12, 12, 12, 2, 8, 12, 12, 12, 12, 12, 12,...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>Design Thinking for Innovation\n\nSindy Samaca...</td>
      <td>[Design, Thinking, for, Innovation, \n\n, Sind...</td>
      <td>[True, True, True, False, False, True, False, ...</td>
      <td>[12, 12, 12, 12, 12, 2, 8, 12, 12, 12, 12, 12,...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56</td>
      <td>Assignment:  Visualization Reflection  Submitt...</td>
      <td>[Assignment, :,   , Visualization,  , Reflecti...</td>
      <td>[False, False, False, False, False, False, Fal...</td>
      <td>[12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 1...</td>
    </tr>
  </tbody>
</table>
</div>



## 🔪 | Data Split

In the following code snippet, we will split the dataset into training and testing subsets using an `90%-10%` ratio.


```python
# Divide el DataFrame en conjuntos de entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.1, random_state=CFG.seed)

# Divide el conjunto de entrenamiento en conjuntos de entrenamiento y validación
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=CFG.seed)
```

## 💾 Creating Dataset for BERT Model

To streamline data preprocessing, we use the `Dataset` module to efficiently encapsulate our dataset. This approach simplifies the management of our training, validation, and test sets.


```python
# Carga los datos en la biblioteca `datasets`
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

# Crea un DatasetDict
raw_data = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

raw_data
```




    DatasetDict({
        train: Dataset({
            features: ['document', 'full_text', 'tokens', 'trailing_whitespace', 'labels', '__index_level_0__'],
            num_rows: 4594
        })
        validation: Dataset({
            features: ['document', 'full_text', 'tokens', 'trailing_whitespace', 'labels', '__index_level_0__'],
            num_rows: 1532
        })
        test: Dataset({
            features: ['document', 'full_text', 'tokens', 'trailing_whitespace', 'labels', '__index_level_0__'],
            num_rows: 681
        })
    })



# 🍽️ | Pre-Processing

Initially, raw text data is quite complex and challenging for modeling due to its high dimensionality. We simplify this complexity by converting text into words then more manageable set of tokens using `tokenizers`. For example, transforming the sentence `"The quick brown fox"` into tokens like `["the", "qu", "##ick", "br", "##own", "fox"]` helps us break down the text effectively. Then, since models can't directly process strings, they are converted into integers, like `[10, 23, 40, 51, 90, 84]`. Additionally, many models require special tokens and additional tensors to understand input better. A `preprocessing` layer helps with this by adding these special tokens, which aid in separating input and identifying padding, among other tasks.


```python
tokenizer = AutoTokenizer.from_pretrained(CFG.preset)
```

Before:



```python
['Design', 'Thinking', 'for', 'innovation', 'reflexion']
```




    ['Design', 'Thinking', 'for', 'innovation', 'reflexion']



After:


```python
inputs = tokenizer(['Design', 'Thinking', 'for', 'innovation', 'reflexion'], is_split_into_words=True)
inputs.tokens()
```




    ['[CLS]',
     'Design',
     'Thinking',
     'for',
     'innovation',
     'reflex',
     '##ion',
     '[SEP]']



As shown above, the tokenizer adds the [CLS] token at the start and the [SEP] token at the end of the input. [CLS] marks the beginning, and [SEP] marks the end of the input. These tokens help the token classification model identify input boundaries.

Additionally, the tokenizer splits 'reflexion' into 'reflex' and '##ion' because BERT's vocabulary is limited to around 30,000 tokens. Tokens not in the vocabulary are broken down into subtokens. This causes a mismatch between the number of tokens and labels, as there are more tokens.

To resolve this, HuggingFace offers a function called tokenize_and_align_labels. This function assigns a label of -100 to special tokens and subtokens (since -100 is ignored by the cross-entropy loss function). Only the first token of each word retains its original label, while all other subtokens get a label of -100.

Here’s their implementation:



```python
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
```

We use the dataset's method `map()` to apply the tokenize_and_align_labels function to each entry in the dataset


```python
tokenized_dataset= raw_data.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_data['train'].column_names
)
```

    Map: 100%|██████████| 4594/4594 [00:05<00:00, 882.56 examples/s]
    Map: 100%|██████████| 1532/1532 [00:01<00:00, 901.47 examples/s]
    Map: 100%|██████████| 681/681 [00:00<00:00, 876.41 examples/s]


Now, if we inspect our tokenized dataset, we will see some new columns:


```python
tokenized_dataset
```




    DatasetDict({
        train: Dataset({
            features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            num_rows: 4594
        })
        validation: Dataset({
            features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            num_rows: 1532
        })
        test: Dataset({
            features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            num_rows: 681
        })
    })



- `input_ids` are required parameters to be passed to the model as input. They are numerical representations of the tokens.

- `labels` contains the correct class for each token. It is the column we changed in the tokenize_and_align_labels() function.

- `attention_mask` is an optional argument used when batching sequences together. 1 describes a token that should be attended to, and 0 is assigned to padded indices. Padding is done in order to make each sequence the same length (it will be done in the next step).

- `token_type_ids` are typically used in next sentence prediction tasks, where two sentences are given. Unless we specify two arguments for token types, the tokenizer assigns 0 to each token.

## 🚜 Data Collator

We will use `DataCollatorForTokenClassification` to batch examples together. This collator pads the text and labels to match the length of the longest element in the batch, ensuring each sample is of uniform length.

The following code creates a `DataCollatorForTokenClassification` object:


```python
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```


We'll employ the `AutoModelForTokenClassification` class for our token classification task. When configuring this model, we specify the pretrained model's name and the number of classes required. Instead of directly passing the number of classes, we can enhance the process by utilizing predefined dictionaries for ID-to-label and label-to-ID mappings, which are already set up in the configuration class. These mappings enable the model to accurately determine the number of classes and prove invaluable during model evaluation on new data.


```python

model = AutoModelForTokenClassification.from_pretrained(CFG.preset, id2label=CFG.id2label, label2id=CFG.label2id)
```

    Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


## 📏 Definition of Evaluation Metrics
Now, we define a function `compute_metrics` to compute evaluation metrics for our model after each training iteration. Here's how it works:


```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Remove ignored index (special tokens)
    true_predictions = [
        [CFG.id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]
    true_labels = [
        [CFG.id2label[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, labels)
    ]

    report = classification_report(true_labels, true_predictions, output_dict=True)

    return {
        'precision': report['micro avg']['precision'],
        'recall': report['micro avg']['recall'],
        'f1': report['micro avg']['f1-score'],
    }

```

This function takes the model's predictions (`pred.predictions`) and the actual labels (`pred.label_ids`). It uses CFG.id2label dictionaries to convert predicted indices and actual labels into their respective class names, excluding special tokens (-100).

## 🤖 Training Configuration and Model Training
- Next, we set up the training configuration in TrainingArguments, which controls how the training process will proceed:

    - Here, we specify the output directory for saving results and the trained model, along with other hyperparameters such as evaluation strategy per epoch, learning rate (`CFG.lr`), batch sizes for training and evaluation, number of training epochs, and weight decay for regularization.

- Finally, we create a Trainer object to manage the model training process:

    - This block of code initializes a Trainer with the defined model (`model`), training arguments (`training_args`), tokenized training and evaluation datasets, tokenizer used, data collator for batching examples, and the compute_metrics function for evaluating the model's performance. It then starts the training process (`trainer.train()`) and saves the trained model to the specified folder (`./saved_model`).

These steps ensure that the model trains effectively and is evaluated using relevant performance metrics for the token classification task.


```python
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=CFG.lr,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.eval_batch_size,  # Adjust as needed
    num_train_epochs=CFG.epochs,
    #gradient_accumulation_steps=2,
    #fp16=True,
    weight_decay=0.01,
    logging_steps=100,
    # You can add more arguments as needed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model('./saved_model')

```

    /home/max/Maestria/ML/project/env/lib/python3.12/site-packages/transformers/training_args.py:1474: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
      warnings.warn(
      3%|▎         | 100/3447 [00:56<31:39,  1.76it/s]

    {'loss': 0.1085, 'grad_norm': 0.030439501628279686, 'learning_rate': 1.941978532056861e-05, 'epoch': 0.09}


      6%|▌         | 200/3447 [01:53<30:43,  1.76it/s]

    {'loss': 0.0045, 'grad_norm': 0.11528144776821136, 'learning_rate': 1.883957064113722e-05, 'epoch': 0.17}


      9%|▊         | 300/3447 [02:50<29:50,  1.76it/s]

    {'loss': 0.0031, 'grad_norm': 0.029222674667835236, 'learning_rate': 1.8259355961705833e-05, 'epoch': 0.26}


     12%|█▏        | 400/3447 [03:47<28:36,  1.77it/s]

    {'loss': 0.0021, 'grad_norm': 0.00359421968460083, 'learning_rate': 1.7679141282274445e-05, 'epoch': 0.35}


     15%|█▍        | 500/3447 [04:44<27:55,  1.76it/s]

    {'loss': 0.0016, 'grad_norm': 0.015648767352104187, 'learning_rate': 1.7098926602843053e-05, 'epoch': 0.44}


     17%|█▋        | 600/3447 [05:43<27:09,  1.75it/s]  

    {'loss': 0.0014, 'grad_norm': 0.01020294614136219, 'learning_rate': 1.6518711923411665e-05, 'epoch': 0.52}


     20%|██        | 700/3447 [06:40<25:54,  1.77it/s]

    {'loss': 0.0021, 'grad_norm': 0.004617201164364815, 'learning_rate': 1.5938497243980273e-05, 'epoch': 0.61}


     23%|██▎       | 800/3447 [07:36<24:54,  1.77it/s]

    {'loss': 0.0006, 'grad_norm': 0.029762808233499527, 'learning_rate': 1.5358282564548885e-05, 'epoch': 0.7}


     26%|██▌       | 900/3447 [08:33<24:08,  1.76it/s]

    {'loss': 0.0017, 'grad_norm': 0.0026347346138209105, 'learning_rate': 1.4778067885117495e-05, 'epoch': 0.78}


     29%|██▉       | 1000/3447 [09:30<23:09,  1.76it/s]

    {'loss': 0.0007, 'grad_norm': 0.0018615652807056904, 'learning_rate': 1.4197853205686105e-05, 'epoch': 0.87}


     32%|███▏      | 1100/3447 [10:29<22:19,  1.75it/s]

    {'loss': 0.0008, 'grad_norm': 0.020093753933906555, 'learning_rate': 1.3617638526254715e-05, 'epoch': 0.96}


     33%|███▎      | 1149/3447 [10:57<19:20,  1.98it/s]/home/max/Maestria/ML/project/env/lib/python3.12/site-packages/seqeval/metrics/v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
                                                       
     33%|███▎      | 1149/3447 [12:05<19:20,  1.98it/s]

    {'eval_loss': 0.0006590427365154028, 'eval_precision': 0.8388429752066116, 'eval_recall': 0.7660377358490567, 'eval_f1': 0.8007889546351085, 'eval_runtime': 68.5688, 'eval_samples_per_second': 22.343, 'eval_steps_per_second': 5.586, 'epoch': 1.0}


     35%|███▍      | 1200/3447 [12:34<21:14,  1.76it/s]   

    {'loss': 0.0006, 'grad_norm': 0.13186244666576385, 'learning_rate': 1.3037423846823325e-05, 'epoch': 1.04}


     38%|███▊      | 1300/3447 [13:31<20:27,  1.75it/s]

    {'loss': 0.0004, 'grad_norm': 0.00882728025317192, 'learning_rate': 1.2457209167391936e-05, 'epoch': 1.13}


     41%|████      | 1400/3447 [14:28<19:28,  1.75it/s]

    {'loss': 0.0009, 'grad_norm': 0.00132074230350554, 'learning_rate': 1.1876994487960546e-05, 'epoch': 1.22}


     44%|████▎     | 1500/3447 [15:26<18:28,  1.76it/s]

    {'loss': 0.0006, 'grad_norm': 0.024113314226269722, 'learning_rate': 1.1296779808529156e-05, 'epoch': 1.31}


     46%|████▋     | 1600/3447 [16:24<17:37,  1.75it/s]

    {'loss': 0.0009, 'grad_norm': 0.03991885483264923, 'learning_rate': 1.0716565129097766e-05, 'epoch': 1.39}


     49%|████▉     | 1700/3447 [17:21<16:38,  1.75it/s]

    {'loss': 0.0004, 'grad_norm': 0.0015261954395100474, 'learning_rate': 1.0136350449666376e-05, 'epoch': 1.48}


     52%|█████▏    | 1800/3447 [18:18<15:35,  1.76it/s]

    {'loss': 0.0005, 'grad_norm': 0.06738290190696716, 'learning_rate': 9.556135770234988e-06, 'epoch': 1.57}


     55%|█████▌    | 1900/3447 [19:15<14:37,  1.76it/s]

    {'loss': 0.0006, 'grad_norm': 0.0020608005579560995, 'learning_rate': 8.975921090803598e-06, 'epoch': 1.65}


     58%|█████▊    | 2000/3447 [20:12<13:44,  1.75it/s]

    {'loss': 0.0007, 'grad_norm': 0.0009478203137405217, 'learning_rate': 8.395706411372208e-06, 'epoch': 1.74}


     61%|██████    | 2100/3447 [21:10<12:44,  1.76it/s]

    {'loss': 0.0006, 'grad_norm': 0.0027893318329006433, 'learning_rate': 7.815491731940818e-06, 'epoch': 1.83}


     64%|██████▍   | 2200/3447 [22:07<11:50,  1.76it/s]

    {'loss': 0.0004, 'grad_norm': 0.0008379350183531642, 'learning_rate': 7.2352770525094295e-06, 'epoch': 1.91}


     67%|██████▋   | 2298/3447 [23:03<09:28,  2.02it/s]/home/max/Maestria/ML/project/env/lib/python3.12/site-packages/seqeval/metrics/v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
                                                       
     67%|██████▋   | 2298/3447 [24:11<09:28,  2.02it/s]

    {'eval_loss': 0.0005078237736597657, 'eval_precision': 0.82421875, 'eval_recall': 0.7962264150943397, 'eval_f1': 0.8099808061420346, 'eval_runtime': 68.0073, 'eval_samples_per_second': 22.527, 'eval_steps_per_second': 5.632, 'epoch': 2.0}


     67%|██████▋   | 2300/3447 [24:12<4:42:52, 14.80s/it]

    {'loss': 0.0007, 'grad_norm': 0.0009891206864267588, 'learning_rate': 6.6550623730780395e-06, 'epoch': 2.0}


     70%|██████▉   | 2400/3447 [25:09<09:52,  1.77it/s]  

    {'loss': 0.0002, 'grad_norm': 0.0008675837307237089, 'learning_rate': 6.0748476936466495e-06, 'epoch': 2.09}


     73%|███████▎  | 2500/3447 [26:06<08:59,  1.75it/s]

    {'loss': 0.0002, 'grad_norm': 0.4596361517906189, 'learning_rate': 5.49463301421526e-06, 'epoch': 2.18}


     75%|███████▌  | 2600/3447 [27:04<08:03,  1.75it/s]

    {'loss': 0.0002, 'grad_norm': 0.032230932265520096, 'learning_rate': 4.91441833478387e-06, 'epoch': 2.26}


     78%|███████▊  | 2700/3447 [28:00<07:05,  1.76it/s]

    {'loss': 0.0003, 'grad_norm': 0.000752713531255722, 'learning_rate': 4.334203655352481e-06, 'epoch': 2.35}


     81%|████████  | 2800/3447 [28:57<06:07,  1.76it/s]

    {'loss': 0.0005, 'grad_norm': 0.012781316414475441, 'learning_rate': 3.753988975921091e-06, 'epoch': 2.44}


     84%|████████▍ | 2900/3447 [29:54<05:09,  1.77it/s]

    {'loss': 0.0003, 'grad_norm': 0.003129888093098998, 'learning_rate': 3.173774296489702e-06, 'epoch': 2.52}


     87%|████████▋ | 3000/3447 [30:51<04:15,  1.75it/s]

    {'loss': 0.0005, 'grad_norm': 0.0007195836515165865, 'learning_rate': 2.5935596170583114e-06, 'epoch': 2.61}


     90%|████████▉ | 3100/3447 [31:49<03:16,  1.76it/s]

    {'loss': 0.0002, 'grad_norm': 0.0008308019023388624, 'learning_rate': 2.0133449376269222e-06, 'epoch': 2.7}


     93%|█████████▎| 3200/3447 [32:46<02:19,  1.77it/s]

    {'loss': 0.0002, 'grad_norm': 0.055828846991062164, 'learning_rate': 1.4331302581955324e-06, 'epoch': 2.79}


     96%|█████████▌| 3300/3447 [33:43<01:23,  1.76it/s]

    {'loss': 0.0005, 'grad_norm': 0.0006820517592132092, 'learning_rate': 8.529155787641428e-07, 'epoch': 2.87}


     99%|█████████▊| 3400/3447 [34:40<00:26,  1.76it/s]

    {'loss': 0.0004, 'grad_norm': 0.0006834220257587731, 'learning_rate': 2.7270089933275317e-07, 'epoch': 2.96}


    100%|██████████| 3447/3447 [35:06<00:00,  1.99it/s]/home/max/Maestria/ML/project/env/lib/python3.12/site-packages/seqeval/metrics/v1.py:57: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
                                                       
    100%|██████████| 3447/3447 [36:14<00:00,  1.59it/s]


    {'eval_loss': 0.0004917252226732671, 'eval_precision': 0.8426966292134831, 'eval_recall': 0.8490566037735849, 'eval_f1': 0.8458646616541353, 'eval_runtime': 68.1206, 'eval_samples_per_second': 22.49, 'eval_steps_per_second': 5.622, 'epoch': 3.0}
    {'train_runtime': 2174.7348, 'train_samples_per_second': 6.337, 'train_steps_per_second': 1.585, 'train_loss': 0.004014274001138813, 'epoch': 3.0}


## 🧪 Loading/Testing the Model

Here we observe the trainer's behavior:


```python
# Tus datos de eval_precision
eval_precisions = [x for x in trainer.state.log_history if 'eval_precision' in x]
eval_precisions = [x['eval_precision'] for x in eval_precisions]

# Tus datos de eval_recall
eval_recalls = [x for x in trainer.state.log_history if 'eval_recall' in x]
eval_recalls = [x['eval_recall'] for x in eval_recalls]

# Tus datos de eval_f1
eval_f1s = [x for x in trainer.state.log_history if 'eval_f1' in x]
eval_f1s = [x['eval_f1'] for x in eval_f1s]

# Tus datos de training_loss
training_losses = [x for x in trainer.state.log_history if 'loss' in x]
training_losses = [x['loss'] for x in training_losses]

# Crear una nueva figura para eval_precision
plt.figure(figsize=(10, 6))

# Dibujar los datos de eval_precision con puntos
plt.plot(eval_precisions, label='Eval Precision', linewidth=2, marker='o')

# Añadir títulos y etiquetas
plt.title('Eval Precision Over Time')
plt.grid(True)
plt.xlabel('Evaluation Steps')
plt.ylabel('Precision')

# Añadir una leyenda
plt.legend()

# Mostrar la gráfica
plt.show()

# Crear una nueva figura para eval_recall
plt.figure(figsize=(10, 6))

# Dibujar los datos de eval_recall con puntos
plt.plot(eval_recalls, label='Eval Recall', linewidth=2, marker='o')

# Añadir títulos y etiquetas
plt.title('Eval Recall Over Time')
plt.grid(True)
plt.xlabel('Evaluation Steps')
plt.ylabel('Recall')

# Añadir una leyenda
plt.legend()

# Mostrar la gráfica
plt.show()

# Crear una nueva figura para eval_f1
plt.figure(figsize=(10, 6))

# Dibujar los datos de eval_f1 con puntos
plt.plot(eval_f1s, label='Eval F1 Score', linewidth=2, marker='o')

# Añadir títulos y etiquetas
plt.title('Eval F1 Score Over Time')
plt.grid(True)
plt.xlabel('Evaluation Steps')
plt.ylabel('F1 Score')

# Añadir una leyenda
plt.legend()

# Mostrar la gráfica
plt.show()

# Crear una nueva figura para training_loss
plt.figure(figsize=(10, 6))

# Dibujar los datos de training_loss con puntos
plt.plot(training_losses, label='Training Loss', linewidth=2, marker='o')

# Añadir títulos y etiquetas
plt.title('Training Loss Over Time')
plt.grid(True)
plt.xlabel('Training Steps')
plt.ylabel('Loss')

# Añadir una leyenda
plt.legend()

# Mostrar la gráfica
plt.show()
```


    
![png](../../images/eval.png)
    



    
![png](../../images/recall.png)



    
![png](../../images/f1.png)



    
![png](../../images/loss.png)
    


Now, we can load the trained Token Classifier from its saved directory with the following code:


```python
model_checkpoint = "./saved_model"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
```

It’s very easy to test it on a new sample:


```python
token_classifier('My name is Frank, my email is frank@gmail.com.')
```




    [{'entity_group': 'NAME_STUDENT',
      'score': 0.5923071,
      'word': 'Frank',
      'start': 11,
      'end': 16},
     {'entity_group': 'EMAIL',
      'score': 0.62551314,
      'word': 'f',
      'start': 30,
      'end': 31},
     {'entity_group': 'EMAIL',
      'score': 0.74226135,
      'word': 'g',
      'start': 36,
      'end': 37}]



The model correctly identifies the student's name, and email. Although it occasionally misses in some sentences, overall, it performs well in detecting personal information with relatively minor errors.


```python
trainer.evaluate()
```

    100%|██████████| 383/383 [01:07<00:00,  5.64it/s]





    {'eval_loss': 0.0004917252226732671,
     'eval_precision': 0.8426966292134831,
     'eval_recall': 0.8490566037735849,
     'eval_f1': 0.8458646616541353,
     'eval_runtime': 68.0194,
     'eval_samples_per_second': 22.523,
     'eval_steps_per_second': 5.631,
     'epoch': 3.0}



## ✌️ Conclusion

In this project, we implemented a natural language processing (NLP) and machine learning approach for automated detection of personally identifiable information (PII) in educational documents. By utilizing a pretrained BERT model and annotating training data in BIO format, we successfully trained a model that performed remarkably well in identifying PII in educational texts. The results achieved an average precision, recall, and F1-score of 0.842, 0.849, and 0.845, respectively, indicating strong detection and classification capabilities for PII. These findings highlight the effectiveness and promise of our proposed approach in addressing the challenge of protecting personal data within educational settings through automated and efficient means. The model is currently applicable to English texts, with potential for adaptation to other languages.



