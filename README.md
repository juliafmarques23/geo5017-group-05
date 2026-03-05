# geo5017-group-05
GEO5017 - Machine Learning (2026) group assignments.

**_Students:_**  
Chaeyeon Moon (6477453)
Evangelia Palli (0000000)
Julia Fossa Marques (655097)

# geo5017-group-05
GEO5017 - Machine Learning (2026) group assignments.

**_Students:_**  
Chaeyeon Moon (6477453)
Evangelia Palli (0000000)
Julia Fossa Marques (6550975)

Provide a ‘ReadMe.txt’ file to briefly explain how to run the code and reproduce the results, e.g.,

dependence on external libraries/packages (includingthe commands for installing them),
the path to data,
where to find the results in case you save results or figures into files ###Descrition
This code was re

The main function was specifically designed to answer the questions of this assignment, using the data provided for it; if the function is called as main, it is not necessary to provide any arguments to reproduce the results shown in the report.

The file also contains the definition for functions constant_velocity, constant_acceleration and plot_trajectoy can be called indepentently from the terminal, or as part of other scripts, with different arguments.

Setup
The program works with minimal setup; the only modules required, besides Python's standard library, are numpy and plotly.

Install numpy (directly from the command prompt)

pyhton -m pip install numpy

Install plotly (directly from the command prompt)

python -m pip install plotly

Parameters:
t : array_like
Timestamps of the position measurements.

positions : array_like
Measured positions in relation to an specific axis. Must have the same lenght as t.

learning_rate : float, optional
Learning rate of the gradient descent algorithm. If it is not specified, the default value (0.001) is used.

Setup
Install numpy

pip install numpy

Install plotly

pip install plotly

Parameters:
t : array_like
Timestamps of the position measurements.

positions : array_like
Measured positions in relation to an specific axis. Must have the same lenght as t.

learning_rate : float, optional
Learning rate of the gradient descent algorithm. If it is not specified, the default value (0.001) is used.

max_iter : int, optional
Maximum number of times the parameters optimization is performed. If it is not specified, the default value (10,000) is used. If it is not specified, the default value (1e-8) is used.*
