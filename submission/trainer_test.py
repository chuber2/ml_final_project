import numpy as np
from sklearn.svm import SVC

# Read in the training data
training_data = np.load('train-resnet-features.npy').item()

# Get the data and label for each training row
num_celebrities = 0
celebrity_labels = {} # A map from a celebrity name to an integer label
celebrity_names = {} # A map from a celebrity label to a name
xTr = []
yTr = []
for img_src in training_data:
  # Get the celebrity name from the image source
  first_slash = img_src.index('/')
  second_slash = img_src.index('/', first_slash + 1)
  celebrity_name = img_src[first_slash+1:second_slash]

  # Get an integer label
  if celebrity_name in celebrity_labels:
    label = celebrity_labels[celebrity_name]
  else:
    label = num_celebrities
    celebrity_labels[celebrity_name] = label
    celebrity_names[label] = celebrity_name
    num_celebrities += 1

  # Add to the training matrices
  xTr.append(training_data[img_src])
  yTr.append(label)

xTr = np.array(xTr)
yTr = np.array(yTr)

# Read in test data
test_data = np.load('test-resnet-features.npy').item()

# Get the labeled validation data
img_srcs = []
xTe = []
for img_src in test_data:
  xTe.append(test_data[img_src])
  img_srcs.append(img_src)
xTe = np.array(xTe)

# Train a classifier
clf = SVC(C=16, kernel='linear')
clf.fit(xTr, yTr)

pred_labels = clf.predict(xTe)

# Write your prediction to the CSV
fout = open('test-submission.csv', 'w')
fout.write('image_label,celebrity_name\n')
for i in range(len(img_srcs)):
  fout.write('{},{}\n'.format(img_srcs[i], celebrity_names[pred_labels[i]]))
fout.close()
