import numpy as np
import pickle
loadedmodel= pickle.load(open('C:/1D/projects/_trainedmodel','rb'))
input_data = (1,128,48,45,194,40.5,0.613,24)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data

prediction = loadedmodel.predict(input_data_reshaped)
print(prediction[0])

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')