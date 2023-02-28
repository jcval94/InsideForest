# De ley
import numpy as np
import pandas as pd
import random
import datetime

# Manejo de docs
import os
import glob
import zipfile

# Data man
import re
import time
import typer
import seaborn as sns

## Necesario para la extracción de los árboles
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt



#ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn import tree
from sklearn.tree import export_text
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import pairwise_distances
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix




