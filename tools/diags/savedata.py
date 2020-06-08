"""
use the pickle module to save data in a file
"""
import numpy as np
import pickle

nz = 10 # ?

data = np.linspace(0, 10, nz)
print(data)

file = "mydata.pkl"
with open(file, "wb") as fid:
    pickle.dump(data, fid)

with open(file, "rb") as fid:
    d = pickle.load(fid)

assert (d == data).all(), "problem: data differ"
print("data read are the same as data written")

#----------------------------------------
# same with more complex data
# pickle accepts any kind of data
# dictionaries are a great way to save several variables at once
data = {"x": np.linspace(0, 10, nz), "title": "some text", "y": np.ones(4)}
file = "mydata2.pkl"
with open(file, "wb") as fid:
    pickle.dump(data, fid)

with open(file, "rb") as fid:
    d = pickle.load(fid)

print("-"*10+" data written")
print(data)
print("-"*10+" data read")
print(d)
