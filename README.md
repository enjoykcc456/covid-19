## COVID-19 Classifier With Transfer Learning

<img src="img/covid-logo.png" class="img-responsive" alt="">

Corona or COVID-19 is a virus that affects human and other mammal’s respiratory system. There are more than 10 millions confirmed cases reported by [WHO](https://covid19.who.int/?gclid=Cj0KCQjwoub3BRC6ARIsABGhnyap-6khHl2aUfbxrKfFFOT2qkw3pCYK8ocAp9Ua4tmsJf4LVTgtKSIaAqQqEALw_wcB) at the time this article is created.

These viruses are actually closely associated with infections like pneumonia, lung inflammation, abscesses, or enlarged lymph nodes. As COVID-19 tests are hard to come by and insufficient to be provided to people all around the world, chest x-ray images are actually one of the important methods to identify the corona virus through imaging.

Hence, the objective of this article would be to create a COVID-19 classifier which will be trained with chest x-ray images and is able to classify them as either positive (COVID-19 infected) or negative (Normal). 

`Note: I am not a medical expert and there are definitely more reliable ways of detecting COVID-19. The main purpose of this article is to walk through the steps to create a classifier with PyTorch by using transfer learning and at the same time exploring how we could apply deep learning knowledge to the medical field.`

Okay. Enough for the intro, let's start!

A few notes before we get started:
1. For simplicity, we will be writing our code and do our training on [Kaggle](https://www.kaggle.com/), utilizing the free GPU provided.
2. [PyTorch](https://pytorch.org/) will be the Deep Learning framework used.
3. There will only be function call throughout this article to avoid cluttering the page. 
The complete code will be available at [Jovian](https://jovian.ml/enjoy-kcc/covid-19-classifier). 
Please note that you will need to sign in to run the code.

Throughout this article, we will step through the following sections to finally get our COVID-19 classifier!
1. Data Analysis
2. Image Data Exploring
3. Data Augmentation
4. Data Loaders
5. Training With Transfer Learning
6. Evaluation With Graph Plotting

### 1. Data Analysis
The data that we are going to use to train our classifier will be the [CoronaHack-Chest X-Ray-Dataset](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset) which is available in Kaggle. We will be exploring and analyzing the data to determine the right data to use for the training. 

First, import all the necessary libraries.
```markdown
import os
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
%matplotlib inline
```
`Please noted that when you run the notebook in Kaggle, you need to click 'Add data' at the top right, search for the dataset and add in into your notebook.
Other than that, you could also start a new notebook with the dataset.`


Defined the constants to access the data and csv files. 
```markdown
DATA_DIR = '../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

METADATA_CSV = '../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv'
```


Read the csv file using pandas and show 10 samples.
```markdown
metadata_df = pd.read_csv(METADATA_CSV)
metadata_df.sample(10)
```
<img src="img/csv-sample.JPG" class="img-responsive" alt="">


Get the size of the original train and test dataset.
```markdown
train_data = metadata_df[metadata_df['Dataset_type'] == 'TRAIN']
test_data = metadata_df[metadata_df['Dataset_type'] == 'TEST']

print(f'Shape of train set: {train_data.shape}')
print(f'Shape of test set: {test_data.shape}')
```
<img src="img/dataset-size.JPG" class="img-responsive" alt="">

Let's us get the amount of null values in each attribute.
```markdown
train_null_vals = train_data.isnull().sum()
test_null_vals = test_data.isnull().sum()
print(f'The number of null values in train set: \n{train_null_vals}\n')
print(f'The number of null values in test set: \n{test_null_vals}')
train_null_vals.plot(kind='bar')
```
<img src="img/null-value.JPG" class="img-responsive" alt="">
<img src="img/null-value-graph.JPG" class="img-responsive" alt="">



You can use the [editor on GitHub](https://github.com/enjoykcc456/covid-19/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```


For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/enjoykcc456/covid-19/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
