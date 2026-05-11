import os
import json
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from bnlp import BengaliWord2Vec, NLTKTokenizer, Word2VecTraining, BengaliCorpus
from tqdm import tqdm
import re
import nltk
nltk.download('punkt_tab')
#from bnlp.tokenizer.basic import BasicTokenizer
#from gensim.models import Word2Vec

# Mount and Extract Dataset
from google.colab import drive
drive.mount('/content/drive')

file_path = '#######'
images_dir = '#######'
caption_json = '#######'

print(f"Image directory length: {len(os.listdir(images_dir))}")
print(f"Caption JSON length: {os.path.getsize(caption_json)}")
