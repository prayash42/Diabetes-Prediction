import numpy as np
import pickle
import streamlit as st
loadedmodel= pickle.load(open('C:/1D/projects/_trainedmodel','rb'))


def diabetes_pred(input_data):
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # standardize the input data

    prediction = loadedmodel.predict(input_data_reshaped)
    print(prediction[0])

    if (prediction[0] == 0):
        return('The person is not diabetic')
    else:
        return('The person is always diabetic')
        
        
def main():
    
    st.title('Diabetes prediction web')
   
    Pregnancies=st.text_input('prerganancy')
    Glucose=st.text_input('Glucose')
    BloodPressure=st.text_input('BloodPressure')
    SkinThickness=st.text_input('SkinThickness')
    Insulin=st.text_input('Insulin')
    BMI=st.text_input('BMI')
    DiabetesPedigreeFunction=st.text_input('numberDiabetesPedigreeFunction')
    Age=st.text_input('age')
        
    
    diagnosis=' '
    
    if st.button('diabetes test result'):
        diagnosis= diabetes_pred([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
    
    


if __name__=='__main__':
    main()




    