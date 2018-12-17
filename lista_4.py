import sys, os
import csv
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


x_matrix = []

first_read = False

with open('reg02.csv', 'rt') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		if first_read == True:
			x_elems = []
			for elem in row:
				x_elems.append(float(elem))
			x_matrix.append(x_elems)
		first_read = True
		

begin_10 = 0
end_10 = 100

for i in range(10):
	x_test = []
	y_test = []    		#destinado aos 10% de teste
	x_training = []
	y_training = []		#destinado a matriz de treino
	x_validation = []
	y_validation = []   #destinado a matriz de validacao
	x_matrix_new = []	#destinado a guardar o que sobrou da matriz ao tirar os 10%

	for j in range(begin_10, end_10):
		y_test.append(x_matrix[j][20])
		x_test.append(x_matrix[j][0:19])

	for j in range(0, begin_10):
		x_matrix_new.append(x_matrix[j])

	for j in range(end_10, 1000):
		x_matrix_new.append(x_matrix[j])

	begin_10 += 100
	end_10 += 100

	for j in range(int(len(x_matrix_new)/2)):
		y_training.append(x_matrix_new[j][20])
		x_training.append(x_matrix_new[j][0:19])

	for j in range(int(len(x_matrix_new)/2), int(len(x_matrix_new))):
		y_validation.append(x_matrix_new[j][20])
		x_validation.append(x_matrix_new[j][0:19])


	min_mse = float('inf')
	min_mae = float('inf')
	k = 1
	mse_array = []
	mae_array = []
	mse_cross_array = []
	mae_cross_array = []

	while True:
		dif = float(0)
		dif_q = float(0)
		regr_1 = DecisionTreeRegressor(max_depth=k)
		regr_1.fit(x_training, y_training)
		predict_val = regr_1.predict(x_validation)
		for i in range(len(predict_val)):
			diference = y_validation[i] - predict_val[i]
			dif_q += diference**2
			dif += abs(diference)
		mse = dif_q/len(predict_val)
		mae = dif/len(predict_val)
		if (mse > min_mse):
			break
		min_mse = mse
		min_mae = mae
		k +=1

	mse_array.append(min_mse)
	mae_array.append(min_mae)

	dif = float(0)
	dif_q = float(0)

	regr_1 = DecisionTreeRegressor(max_depth=k-1)
	regr_1.fit(x_training, y_training)
	predict_cross_validation = regr_1.predict(x_test)
	for i in range(len(predict_cross_validation)):
		diference = y_test[i] - predict_val[i]
		dif_q += diference**2
		dif += abs(diference)
	
	mse = dif_q/len(predict_val)
	mae = dif/len(predict_val)

	mse_cross_array.append(mse)
	mae_cross_array.append(mae)

mean_mse = 0
mean_mse_cross = 0
mean_mae = 0
mean_mae_cross= 0

for i in range(len(mse_array)):
	mean_mse += mse_array[i]
	mean_mae += mae_array[i]
	mean_mse_cross += mse_cross_array[i]
	mean_mae_cross += mae_cross_array[i]

mean_mse = mean_mse/len(mse_array)
mean_mae = mean_mae/len(mse_array)
mean_mse_cross = mean_mse_cross/len(mse_array)
mean_mae_cross = mean_mae_cross/len(mse_array)

print("MSE base de treino: " + str(mean_mse) + "; MAE base de treino: " + str(mean_mae))
print("MSE base de validação: " + str(mean_mse_cross) + "; MAE base de validação: " + str(mean_mae_cross))

