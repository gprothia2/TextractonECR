import os
import shutil

directory = './input/'
for filename in os.listdir(directory):
    for i in range(100):
      dest = './input/'+str(i) + filename
      print(dest)
      shutil.copy(directory+filename,dest)
    print(filename)
