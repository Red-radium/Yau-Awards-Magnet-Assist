import pandas as pd
from renderer import *

df = pd.read_csv("trajectory.csv", header=None, names=["t", "x", "y", "z"])

T = df["t"].values
Pos = df[["x", "y", "z"]].values

render_trajectory(T,Pos)
