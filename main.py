import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
#pillow 9.0.1

# a = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
# print(a)
# print(a[0][0])
# print(a[1][1])
# print(a[2][2])
#
# a = np.arange(12).reshape((4, 3)) #macierz 4 na 3
# print(a)
# print(a.sum())
# print(a.sum(axis = 0))
# print(a.sum(axis = 1)) #1 - wiersze macierzy
# print(a.cumsum(axis = 1)) #suma skumulowanych elementow
#
# b = np.array([20, 30, 40, 50])
# c = np.arange(4)
#
# d = c + b
# print("c + b = ", d)
# e = np.sqrt(c)
# f = d + e
# print(f) #upcasting

# a = np.array([[2, 5, 1], [5, 7, 1]])
# b = np.array([[2, 3], [4, 5], [7,1]])
# c = np.dot(a, b)
# print(c)
# d = a.dot(b)
# print(d)
#
# e = np.arange(3)
# f = np.arange(3)
# print(np.dot(e, f))
# g = np.arange(3)
# h = np.array([[0], [1], [2]])
# print(g.dot(h))

# a = np.arange(6).reshape((3, 2))
# print(a.flat) #flat - kazdy element oddzielnie
# # for b in a.flat:
# #     print(b)
# c = a.reshape((2, 3))
# print(c)
# b = c.ravel()
# print(b)
# d = c.T
# print(d)

s = pd.Series((1, 3, 4, 'a', 3.5))
print(s)
s = pd.Series([10, 12, 14, 8], index = ['a', 'b', 'c', 'd'])
print(s)

data = {'Kraj': ['Belgia', ' Indie', 'Brazylia', 'Polska'],
        'Stolica': ['Bruksela', 'New Delhi', 'Brasillia', 'Warszawa'],
        'Populacja': [11190826, 1303171035, 207847528, 38675467],
        'Kontynent': ['Europa', 'Azja', 'Ameryka Poludniowa', 'Europa']}
df = pd.DataFrame(data)
# print(df)
#
daty = pd.date_range('20220507', periods = 5)
# print(daty)
df2 = pd.DataFrame(np. random.rand(5, 4), index = daty, columns = list('ABCD'))
# print(df2)
#
df3 = pd.read_csv('dane.csv', header = 0, sep = ';', decimal = '.')
print(df3)
# df3.to_csv('dane2.csv', index = False)
# #openpyxl
# xlsx = pd.ExcelFile('imiona.xlsx')
# df = pd.read_excel(xlsx, header = 0)
# print(df)
# df.to_excel('imiona2.xlsx', sheet_name = 'dane')

# print(s['a']) #odwolanie sie do serii danych
# print(s.a)
#
# print(df[0:1])
# print(df['Kraj'])
# print(df.Kraj)
#
# print(df.iloc[[0], [0]])
# print(df.loc[[0], ['Kraj']])
# print(df.at[0, 'Kraj'])

# print(df.sample(2))
# print(df.sample(10, replace = True)) #losowanie 10 wierszy z powtarzaniem

# print(df4.head(10))
# print(df4.tail(10))

# print(s['a']) #odwolanie sie do serii danych znowu : )
# print(s.a)
#
# print(s[s > 8])
# print(s[(s < 13) & (s > 8)])
#
# print(s.where(s > 10, 'warunek niespelniony'))
# seria = s.copy()
# seria.where(s > 10, 'warunek niespelniony', inplace = True)
#
# print(df[df['Populacja'] > 12000000])
#
# s['e'] = 14
# print(s)
#
# df.loc[3] = 'nowy element'
# print(df)
# df.loc[4] = ['Polska', 'Warszawa', 386754468]
# print(df)
#
# df.drop([3], inplace = True)
# print(df)
# # print(new_df)
# df['Kontynent'] = ['Europa', 'Azja', 'Ameryka Poludniowa', 'Europa']
# print(df)
#
# print(df.sort_values(by = 'Kraj'))
# new = df.sort_values(by = 'Kraj', ascending = False)
# print(new)
#
# grupa = df.groupby(by = 'Kontynent').agg({'Populacja': ['sum']})
# print(grupa.get_group('Europa'))
# print(grupa.agg({'Populacja': ['sum']})) #albo ta agregacja tutaj albo ta 2 linijki wyzej!

# matplotlib 3.5.1
# grupa.plot(kind = 'bar', xlabel = 'Kontynenty', ylabel = 'Populacja w mld', rot = 0, title = 'Populacja na kontynentach')
# wykres = grupa.plot.bar()
# wykres.set_xlabel('Kontynent')
# wykres.set_ylabel('Populacja w mld')
# wykres.tick_params(axis = 'x', labelrotation = 0)
# wykres.set_title('Populacja na kontynentach')
# plt.savefig('plot1.png')
# plt.show()

# grupa = df3.groupby('Imię i nazwisko').agg({'Wartość zamówienia': ['sum']})
# print(grupa)
# grupa.plot(kind='pie', subplots=True, autopct='%.2f %%', fontsize=20, colors=['red', 'green'])
# plt.legend(loc='upper left')
# plt.show()

# seria = pd.Series(np.random.randn(1000))
# seria = seria.cumsum()
#
# seria.plot()
# plt.show()

#21.05.22
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro:') #r - red o - kropka : - linia kropkowana
# plt.ylabel("liczby z wektora")
# plt.show()
#
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r:')
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'bo')
#
# plt.axis([0, 6, 0, 20])
# plt.show()

x = np.arange(0, 5.2, 0.2)

# plt.plot(x, x, 'r-', x, x**2, 'b^', x, x**3, 'gs')
# plt.legend(labels=['liniowa', 'kwadratowa', 'szescienna'], loc='center left')

# plt.plot(x, x, label='liniowa')
# plt.plot(x, x**2, label='kwadratowa')
# plt.plot(x, x**3, label='szescienna')
# plt.legend()
# plt.savefig('plot.png')
# plt.show()
# im1 = Image.open('plot.png')
# im1 = im1.convert('RGB')
# im1.save('plot.jpg')
#
# x = np.arange(1, 21, 1)
# plt.plot(x, 1/x, 'r-',label="funkcja homograficzna")
# plt.legend()
# plt.show()
#
# x = np.arange(0, 11, 0.1)
# plt.plot(x, np.sin(x), 'r-', label="sin od x")
# plt.legend()
# plt.show()

x1 = np.arange(0, 2.02, 0.02)
x2 = np.arange(0, 2.02, 0.02)

y1 = np.sin(2*np.pi*x1)
y2 = np.cos(2*np.pi*x2)

# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'r--')
# plt.ylabel('sin(y)')
# plt.xlabel('x')
# plt.title('wykres sin(x)')
#
# plt.subplot(2, 1, 2) #2 wiersze 1 kolumna indeks 2
# plt.plot(x2, y2, 'go')
# plt.xlabel('x')
# plt.ylabel('cos(y)')
# plt.title('wykres cos(x)')
# plt.subplots_adjust(hspace=0.5)
# plt.show()

# fig, axs = plt.subplots(3, 2)
# print(type(fig))
# print(type(axs))
#
# axs[0, 0].plot(x1, y1, 'r-')
# axs[0, 0].set_xlabel('x')
# axs[0, 0].set_ylabel('sin(x)')
# axs[0, 0].set_title('wykres sin(x)')
#
# axs[1, 1].plot(x1, y1, 'r-')
# axs[1, 1].set_xlabel('x')
# axs[1, 1].set_ylabel('cos(x)')
# axs[1, 1].set_title('wykres cos(x)')
#
# axs[2, 0].plot(x2, y2, 'g-')
# axs[2 ,0].set_xlabel('x')
# axs[2 ,0].set_ylabel('cos(x)')
# axs[2 ,0].set_title('wykres cos(x)')
# fig.delaxes(axs[0 ,1])
# fig.delaxes(axs[1 ,0])
# fig.delaxes(axs[2 ,1])
# plt.show()
#
# dane = {'a': np.arange(50),
#         'c': np.random.randint(0, 50, 50),
#         'd': np.random.randn(50)}
#
# dane['b'] = dane['a'] + 10 * np.random.randn(50)
# dane['d'] = np.abs(dane['d']) * 100
#
# plt.scatter(data=dane, x='a', y='b', color='red', s='d')
# plt.xlabel('wartosci a')
# plt.ylabel('wartosci b')
# plt.show()
# print(dane['c'])

# df = pd.DataFrame(data)
# print(df)
# grupa = df.groupby('Kontynent')
# etykiety = list(grupa.groups.keys())
# wartosc = list(grupa.agg('Populacja').sum())
#
# plt.bar(x=etykiety, height=wartosc, color=['red', 'green', 'blue'])
# plt.xlabel('Kontynenty')
# plt.ylabel('Populacja na kontynentach')
# plt.show()

x = np.random.randn(10000)

plt.hist(x, bins=15, facecolor='g', alpha=0.75, density=True)
plt.xlabel('wartosci x')
plt.ylabel('prawdopodobienstwa')
plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# #pillow 9.0.1

# x = np.arange(-2, 4, 0.12)
# plt.subplot(2, 1, 1)
# plt.plot(x, ((-4)*(np.power(x, 2)) + ((6 * x) / 2) + 20), 'ro', label="sin od x")
# plt.legend(labels=['-4*x^2+(6x/2)+20'], loc='lower center')
# plt.grid()
# plt.axis([-2, 4, -25, 25])
# plt.yticks([-25, 0, 25])
# plt.show()
