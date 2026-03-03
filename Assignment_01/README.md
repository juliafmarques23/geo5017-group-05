### Setup

Install numpy

> pip install numpy

Install plotly

> pip install plotly

### **Parameters:**

**t** : array_like  
*Timestamps of the position measurements.*

**positions** : array_like  
  *Measured positions in relation to an specific axis. Must have the same lenght as* t.

**label** : string  
  *Axis for which the measuraments were made* (e.g. "X").

**start_p0** : float, optional  
  *Starting value for the initial position.
  If it is not specified, the default value (0.0) is used.*

**start_v** : float, optional  
  *Starting value for the velocity.
  If it is not specified, the default value (0.0) is used.*

**learning_rate** : float, optional  
  *Learning rate of the gradient descent algorithm. 
  If it is not specified, the default value (0.001) is used.*

**max_iter** : int, optional  
  *Maximum number of times the parameters optimization is performed.
  If it is not specified, the default value (10,000) is used.*

**tolerance** : float, optional  
  *Minimum step size for parameter optimization.
  If it is not specified, the default value (1e-8) is used.*
  
