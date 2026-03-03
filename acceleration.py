import numpy as np

#SAMPLE POINTS:

Px = [2, 1.08, -0.83, -1.97, -1.31, 0.57]
Py = [0, 1.68, 1.82, 0.28, -1.51, -1.91]
Pz = [1, 2.38, 2.49, 2.15, 2.59, 4.32]
time = [1, 2, 3, 4, 5, 6]

def gradient_alg(samplepositions=type[list], timestamps=type[list], p0=type[float], v0=type[float], a=type[float], learning_rate=0.01, tolerance=0.001):

    nsamples = len(samplepositions)
    sum_squared_errors = 0
    dv0 = 0     #partial derivative with respect to Vo
    dp0 = 0     #partial derivative with respect to Po
    da = 0     #partial derivative with respect to a

    for i in range(nsamples):

        position = samplepositions[i]
        t = timestamps[i]
        predicted = p0 + v0*t + (a/2) * t**2

        squared_error = (position - predicted)**2
        sum_squared_errors += squared_error

        # iteratively sum the partial derivatives
        # their average will be used to calculate the next step size
        da += -2 * (position - predicted) * t**2 / 2
        dv0 += -2 * (position - predicted) * t
        dp0 += -2 * (position - predicted)

    #compute the size of the next step:
    dv0 = dv0/nsamples
    dp0 = dp0/nsamples
    da = da/nsamples
    stepv = learning_rate * dv0
    stepp = learning_rate * dp0
    stepa = learning_rate * da

    #check if the step is bigger than the minimum step parameter
    #if abs(stepv) < tolerance or abs(stepp) < tolerance:
    #store the current values for the print statement
    #update the values of v0 and p0, to be used in the next iteration
    oldv0 = v0
    oldp0 = p0
    olda = a
    v0 -= stepv
    p0 -= stepp
    a -= stepa

    return v0, p0, a, sum_squared_errors, oldv0, oldp0, olda

def estimate_params(y, x, a, b, c, learning_rate=0.01, tolerance=0.001, maxsteps=1000, dimension=type[str]):
    po = c
    vo = b
    ae = a
    for i in range(maxsteps):
        Vo, Po, ca, error, oldvo, oldpo, olda = gradient_alg(samplepositions=y, timestamps=x, p0=po, v0=vo, a0=ae, learning_rate=learning_rate, tolerance=tolerance)
        print("Iteration {0} -> Po{1} = {2}, Vo{3} = {4}, a = {5}, sum squared erros = {6}".format(i+1, dimension, oldpo, dimension, oldvo, olda, error))
        po = Po
        vo = Vo
        ae = ca

    estimated_positions = np.zeros(shape=len(x))
    for d in range(len(x)):
        estimated_position = po + vo * x[d] + (a/2)*x[d]**2
        estimated_positions[d] = estimated_position

    return po, vo, ae, estimated_positions

#compute final estimated positions:

pox, vox, ax, estimated_x = estimate_params(x=time, y=Px, a=0, b=0, c=0, dimension="X")
poy, voy, ay, estimated_y = estimate_params(x=time, y=Py, a=0, b=0, c=0, dimension="Y")
poz, voz, az, estimated_z = estimate_params(x=time, y=Pz, a=0, b=0, c=0, dimension="Z")
estimated_coords = np.vstack((estimated_x, estimated_y, estimated_z))

estimated_positions = np.transpose(estimated_coords)

print(estimated_positions)