import numpy as np

#SAMPLE POINTS:

Px = [2, 1.08, -0.83, -1.97, -1.31, 0.57]
Py = [0, 1.68, 1.82, 0.28, -1.51, -1.91]
Pz = [1, 2.38, 2.49, 2.15, 2.59, 4.32]
time = [1, 2, 3, 4, 5, 6]

def gradient_alg(samplepositions=type[list], timestamps=type[list], p0=type[float], v0=type[float], learning_rate=0.01, tolerance=0.001):

    nsamples = len(samplepositions)
    sum_squared_errors = 0
    dv0 = 0     #partial derivative with respect to Vo
    dp0 = 0     # partial derivative with respect to Po
    stepp = 1
    stepv = 1

    for i in range(nsamples):

        position = samplepositions[i]
        t = timestamps[i]
        predicted = p0 + v0*t

        squared_error = (position - predicted)**2
        sum_squared_errors += squared_error

        # iteratively sum the partial derivatives
        # their average will be used to calculate the next step size
        dv0 += -2 * (position - predicted) * t
        dp0 += -2 * (position - predicted)

    #compute the size of the next step:
    dv0 = dv0/nsamples
    dp0 = dp0/nsamples
    stepv = learning_rate * dv0
    stepp = learning_rate * dp0

    #check if the step is bigger than the minimum step parameter
    #if abs(stepv) < tolerance or abs(stepp) < tolerance:
    #store the current values for the print statement
    #update the values of v0 and p0, to be used in the next iteration
    oldv0 = v0
    oldp0 = p0
    v0 -= stepv
    p0 -= stepp

    return v0, p0, sum_squared_errors, oldv0, oldp0

def estimate_params(y, x, a, b, learning_rate=0.01, tolerance=0.001, maxsteps=1000, dimension=type[str]):
    po = a
    vo = b
    for i in range(maxsteps):
        Vo, Po, error, oldvo, oldpo = gradient_alg(samplepositions=y, timestamps=x, p0=po, v0=vo,learning_rate=learning_rate, tolerance=tolerance)
        print("Iteration {0} -> Vo{1} = {2}, Po{3} = {4}, sum squared erros = {5}".format(i+1, dimension, oldvo, dimension, oldpo, error))
        po = Po
        vo = Vo

    estimated_positions = np.zeros(shape=len(x))
    for d in range(len(x)):
        estimated_position = po + vo * x[d]
        estimated_positions[d] = estimated_position

    return po, vo, estimated_positions

#compute final estimated positions:

pox, vox, estimated_x = estimate_params(x=time, y=Px, a=0, b=0, dimension="X")
poy, voy, estimated_y = estimate_params(x=time, y=Py, a=0, b=0, dimension="Y")
poz, voz, estimated_z = estimate_params(x=time, y=Pz, a=0, b=0, dimension="Z")
estimated_coords = np.vstack((estimated_x, estimated_y, estimated_z))

estimated_positions = np.transpose(estimated_coords)

print(estimated_positions)