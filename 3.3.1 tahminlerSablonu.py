# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('maaslar.csv')
    
x= veriler.iloc[:,1:2]  ## eğitim seviyesi  DATAFRAFME BİLGİLERİ   AMAC UNVANI ATMAK ISE YARAMIYOR SUANLIK  
y=veriler.iloc[:,2:]   ## maaslar 
X=x.values
Y=y.values  ## KUCUK X DATAFRAME BÜYÜK X VE Y NUMPY ARRAY OLMASIDIR 

## bu bir polinomal regresyon ama biz bi lineer de bakalım nasıl oluyormus
from sklearn.linear_model import LinearRegression
lin_reg1=LinearRegression()
lin_reg1.fit(X,Y)  ## x ten y yi ögren  

## görselleştirmek için

plt.scatter(X,Y , color='b') ## 2 boyutlu x ve y oldugunu soyluyoruz
plt.plot(X,lin_reg1.predict(X),color = 'g')  ## predict olarak her bir x karşılık gelen tahminleri görselleştirecez
plt.show()
print('linear R2 degeri')
print(r2_score(Y, lin_reg1.predict(X)))

## POLİNOMAL REGRESSİON

from sklearn.preprocessing import PolynomialFeatures ## herhangi bir sayıyı polinaml ifade etmeye yarıyor
poly_reg = PolynomialFeatures(degree=4) ## polinomal 2 dereceden olustur bir obje 

x_poly = poly_reg.fit_transform(X)
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly, y)##   x poliyi y ye göre fit et y yi ögren

plt.scatter(X,Y , color = 'red')
plt.plot(X , lin_reg2.predict(poly_reg.fit_transform(X)), color = 'b')  ## Polinomal domain e cevirmemiz lazım
plt.show()



#TAHMİN
print(lin_reg1.predict([[9]]))  ## lineer le tahmin yaptık
print(lin_reg1.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))  ## poly ile tahmin yaptık
print(lin_reg2.predict(poly_reg.fit_transform([[10]])))

print('Polynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))  ## POLİNOMAL OLDUGU İÇİN TRANSFORM KULLANDIK

#veri ölçeklendirme
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()  ## bunu yapmalıyız standart almaylıyız 
x_olcekli = sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=np.ravel(sc2.fit_transform(Y.reshape(-1,1)))  ## ?? 


## SVR 
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')  ## kernel secicegimiz model rpf en mantıklı olanı 

svr_reg.fit(x_olcekli, y_olcekli)
plt.scatter(x_olcekli,y_olcekli)
plt.plot(x_olcekli, svr_reg.predict(x_olcekli))
plt.show()
print('SVM için R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


'''
print (svr_reg.predict([11]))
'''
## Karar Agacı ( Decision Tree)
from sklearn.tree import DecisionTreeRegressor

r_dt=DecisionTreeRegressor(random_state=0)
Z=X+0.5
K=X-0.4

r_dt.fit(X,Y) ## NP ARRAYLARINI ÖGREN X TEN Y Yİ ÖGREN 
plt.scatter(X,Y)  ## Bu ikisini arasındakı uzayı çiz
plt.plot(X,r_dt.predict(X)) ## ,


plt.show()
print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))
'''
plt.plot(x,r_dt.predict(Z),color='red')
plt.plot(x, r_dt.predict(K),color = 'brown')
'''

## Random forest Regresyonu ##
## ensemble = birden falza kısıden olusan bir grup gibi dusun
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0) ## estimators 10 demek 10 tane karar agacı  tree ciz demek
rf_reg.fit(X,Y.ravel()) ## ravel ? 

 
plt.scatter(X, Y, color='red')
plt.plot(X,rf_reg.predict(X),color='blue')  ## verilen x degeri için tahmin x degerini ciz diyoruz 

print('Random forest 6.5 indexinde tahmin edilen :'  )
print(rf_reg.predict([[6.5]]))


from sklearn.metrics import r2_score

print('Random forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))
plt.show()


## ÖZET R2 Degerleri
print('-----------------------------')
print('linear R2 degeri')
print(r2_score(Y, lin_reg1.predict(X)))

print('Polynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))  ## POLİNOMAL OLDUGU İÇİN TRANSFORM KULLANDIK

print('SVM için R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))

print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))

print('Random forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))










