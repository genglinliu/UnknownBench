
import os
import time

for filename in os.listdir("."):
    if "_response" in filename:
        # replace "out" with "answerable"
        new_filename = filename.replace("_response", "")
        os.rename(filename, new_filename)