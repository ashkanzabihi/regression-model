import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


product = {
    'production_hours': [29, 9, 10, 38, 16, 26, 50, 10, 30, 33, 43],
    'production_yield': [65, 7, 8, 76, 23, 56, 100, 3, 74, 48, 73]
}


product_data = pd.DataFrame(data=product)

x = product_data.production_hours
y = product_data.production_yield

plt.scatter(x,y, label="Data poit", color = 'blue')
plt.xlabel("production_hours")
plt.ylabel("production_yield")
plt.title("Production_Yield w.r.t Production_Hours")
plt.grid()
plt.legend(['Data Points'])
plt.show()

model = np.polyfit(x,y,1)
model

print(model)

predict = np.poly1d(model)

production_hours = 20
production_yield_Result = predict(production_hours)
production_yield_Result


from sklearn.metrics import r2_score
r2_score(y, predict(x))

#print(r2_score(y, predict(x)))

x_lin_reg = range(0,51)
y_lin_reg = predict(x_lin_reg)
plt.scatter(x,y)
plt.grid()
plt.plot(x_lin_reg, y_lin_reg, c = 'b')
plt.show()
