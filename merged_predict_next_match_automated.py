# By Edmundo Cuadra

import pandas as pd # For data manipulation
import numpy as np # For data computation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import os
import requests
import math
from collections import defaultdict
from datetime import datetime
import streamlit as st
import plotly.graph_objs as go
import bs4
from bs4 import BeautifulSoup
import altair as alt

st.header("Season's Over!")
st.subheader("Stay Tuned for the Next One!")


