# Measured positions:

x = []
y = []
z =[]
positions = [(2, 0, 1), (1.08, 1.68, 2.38),
                 (-0.83, 1.82, 2.49), (-1.97, 0.28, 2.15),
                 (-1.31, -1.51, 2.59), (0.57, -1.91, 4.32)]

for position in positions:
    x.append(position[0])
    y.append(position[1])
    z.append(position[2])

tl = [1, 2, 3, 4, 5, 6]

# Sums and averages for X, Y and Z positions, and time

sumx = sum(x)
sumy = sum(y)
sumz = sum(z)
sumt = sum(tl)

avgx = sumx/len(x)
avgy = sumy/len(y)
avgz = sumz/len(z)
avgt = sumt/len(tl)

# Variance of time
sumt2 = 0
for val in tl:
    i = (val-avgt)**2
    sumt2 += i

variancet = sumt2/(len(tl)-1)

# ESTIMATION OF X POSITIONS

#variance of X position
sumx2 = 0
for val in x:
    i = (val-avgx)**2
    sumx2 += i
variancex = sumx2/(len(x)-1)

#covariance(t, X)
sum_xt = 0
t = 0
for pos in x:
    i = (pos-avgx) * (t-avgt)
    sum_xt += i
    t += 1
cov_xt = sum_xt/(len(x)-1)

#Velocity in the X dimension and initial X position
Vx = cov_xt/variancet
Xo = avgx - (Vx * avgt)
print("Vx = ", Vx)
print("Xo = ", Xo)

#Estimation of X positions
ex = []
for time in tl:
    i = Xo + (Vx*time)
    ex.append(i)
print("Estimated X positions = ", ex)

# Sum of squares of errors for x:
sumex = 0
for i in range(0,6):
    a = (ex[i] - x[i])**2
    sumex += a
print("Sum of squares errors for x = ", sumex)

#ESTIMATION OF Y POSITIONS

# Variance of Y positions
sumy2 = 0
for val in y:
    i = (val-avgy)**2
    sumy2 += i
variancey = sumy2/(len(y)-1)

#covariance (t, y)
sum_yt = 0
t = 0
for pos in y:
    i = (pos-avgy) * (t-avgt)
    sum_yt += i
    t += 1
cov_yt = sum_yt/(len(y)-1)

#Velocity in the Y dimension and initial Y position
Vy = cov_yt/variancet
Yo = avgy - (Vy * avgt)
print("Vy = ", Vy)
print("Yo = ", Yo)

#Estimation of Y positions
ey = []
for time in tl:
    i = Yo + (Vy*time)
    ey.append(i)
print("Estimated Y positions = ", ey)

# Sum of squares of errors for y:
sumey = 0
for i in range(0,6):
    a = (ey[i] - y[i])**2
    sumey += a

print("Sum of squares of errors for y = ", sumey)

#ESTIMATION OF Z POSITION

#variance of Z positions
sumz2 = 0
for val in z:
    i = (val-avgz)**2
    sumz2 += i
variancez = sumz2/(len(z)-1)

# Covariance (t, z)
sum_zt = 0
t = 0
for pos in z:
    i = (pos-avgz) * (t-avgt)
    sum_zt += i
    t += 1
cov_zt = sum_zt/(len(z)-1)

# Velocity in the Z dimension and initial Z position
Vz = cov_zt/variancet
Zo = avgz - (Vz * avgt)
print("Vz = ", Vz)
print("Zo = ", Zo)

#Estimation of Y positions
ez = []
for time in tl:
    i = Zo + (Vz*time)
    ez.append(i)
print("Estimated Z positions = ", ez)

# Sum of squares of errors for x:
sumez = 0
for i in range(0,6):
    a = (ez[i] - z[i])**2
    sumez += a
print("sum square errors for z = ", sumez)

# ESTIMATED POSITIONS
estimated_positions = []
for i in range(0,6):
    p = (ex[i], ey[i], ez[i])
    estimated_positions.append(p)
print("Estimated positions = ", estimated_positions)
