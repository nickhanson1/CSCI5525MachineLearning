Nick Hanson
5458741
hans6064@umn.edu

How to run: SVM_dual.py
  python SVM_dual.py <filename>
    where <filename> is the path to the dataset

How to run: kernel_SVM.py
  python kernel_SVM.py <filename>
    where <filename> is the path to the dataset

How to run: multi_SVM.py
  python multi_SVM.py <filename>
    where <filename> is the path to the dataset
    OR where <filename> is the string "all", in which case
    multi_SVM looks into the "mfeat/" folder and uses all features of the mfeat dataset at once

Note in each program only 20 percent of the data given was used to save on runtime, however these lines are commented out, and so each SVM runs as intended.

The "all" option of multi_SVM must have the mfeat data in a folder "mfeat/" in the same directory as multi_SVM.py
