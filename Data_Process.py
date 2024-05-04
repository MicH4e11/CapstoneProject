import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class CarPredictionApp:

    def __init__(self, master):
        self.master = master
        self.master.title('Car Price Prediction')
        
        #First read the data from Numeric file
        self.data = pd.read_pickle('DataForML_Numeric.pkl')
        self.Predictors = ['Fuel_Type', 'Transmission', 'Year', 'Power (bhp)']
        self.TargetVariable = 'Price'
        #The numeric file is then divided into predictors and price file as X and Y
        self.X = self.data.drop('Price', axis=1)
        self.X = self.X[self.Predictors].values
        self.Y = self.data['Price'].values

        #Split training for the model
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=1000)
        
        #The model uses the same parameters and no normalisation since we're not using KNN or deep learning for this
        self.model = XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=1000, objective='reg:linear', booster='gbtree')
        self.model.fit(self.X_train, self.Y_train)
        self.create_widgets()


    def create_widgets(self):
        self.sliders = []
        self.combo = {}
        #This is to store the categorical data for creating the combo box
        self.Cat_Literal = {'Owner_Type':['First', 'Second','Third','Fourth & Above'],
                          'Transmission': ['Manual', 'Automatic'],
                          'Fuel_Type': ['Diesel', 'Petrol','CNG','LPG', 'Electric'],
                          'Year': [1998, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]}

        #For each column and cell inside data
        for i, column in enumerate(self.data.columns):
            if column in self.Predictors:
                #If it's power, which is the only continuous data inside the predictors
                #We make a slider
                if column == 'Power (bhp)':
                    label = tk.Label(self.master, text=column + ': ')
                    label.grid(row=i, column=0)
                    current_val_label = tk.Label(self.master, text='0.0')
                    current_val_label.grid(row=i, column=2)
                    #creating the slider with min/max value from the min/max of the power column
                    self.Pow_slider = ttk.Scale(self.master, from_=self.data[column].min(), to=self.data[column].max(), orient="horizontal",
                                    command=lambda val, label=current_val_label: label.config(text=f'{float(val):.2f}'))
                    self.Pow_slider.grid(row=i, column=1)
                else:
                    label = tk.Label(self.master, text=column + ': ')
                    label.grid(row=i, column=0)
                    combo_box = ttk.Combobox(self.master, values=self.Cat_Literal[column]) #create combobox folowwing Categorical Literal dictionary
                    combo_box.current(0) #set combobox default value as first one 
                    combo_box.grid(row=i, column=1)
                    self.combo[column] = combo_box
        #Otherwise, we make 
        predict_button = tk.Button(self.master, text='Predict Price', command=self.predict_price)
        predict_button.grid(row=len(self.data.columns), columnspan=3)



    def Combo_Convert(self):
        #This is to store the categorical data for converting into data for processing
        self.Cat_Convert = {  'Owner_Type': {'First': 0, 'Second': 1,'Third':2,'Fourth & Above': 3},
                                'Transmission': {'Manual': 0, 'Automatic': 1},
                                'Fuel_Type': {'Diesel': 0, 'Petrol': 1,'CNG':2,'LPG':3, 'Electric': 4},
                            }
        output = []

        for column in self.combo:
            if column != 'Year':
                #get value of the combo box
                value = self.combo[column].get()
                #convert it using Cat_convert dictionary
                value = self.Cat_Convert[column][value]
                #add it to the combo value
                output.append(value)
            else:
                #If it's year, no need to convert, just 
                value = int(self.combo[column].get())
                output.append(value)
        return output
        
    def predict_price(self):
        if self.Pow_slider.get() != 0.0:
            price = 0
            inputs = self.Combo_Convert() #get converted value from combo box
            inputs.append(float(self.Pow_slider.get())) #get the power value from slider
            print(inputs) #print the inputs for debugging
            price = self.model.predict([inputs], [])
            price = np.round(  np.exp(price) - 1 , decimals= 2)
            real_price = np.round(price * 100000)

            messagebox.showinfo('Predicted Price', f'The predicted price is {price[0]:.2f} Lakh = {real_price[0]:.0f} rupees')
        else:
            #if power is invalid(=0), print message
            messagebox.showinfo('Invalid','Invalid power scale, try again')



if __name__ == '__main__':
    root = tk.Tk()
    app = CarPredictionApp(root)
    root.mainloop()
