# University Final year project: Evolutionary Algorithms code
A project for solving the Teaching Allocation problem using NSGA-III, the state-of-the-art multi-objective evolutionary algorithm. This was the subject of my final year project at the University of Birmingham's Artificial Intelligence and Computer Science course.
## File descriptions
- `main.py`: Starting point for all experiments, containing calls to all other files.
- `generateData.py`: Creates the `Data` class instance used for every test of the Teaching Allocation problem
- `NSGA3.py`: Runs the NSGA-III algorithm according to the parameters specified when called. 
- `NSGA2.py`: Runs the NSGA-II algorithm according to the parameters specified when called. 
- `MOEAD.py`: Runs the MOEA/D algorithm according to the parameters specified when called. 
- `recursiveParentoShellWithDuplicates.py`: Contains shell-sorting function used in all evolutionary algorithms.
- `supportFunctions.py`: Contains support functions used in all evolutionary algorithms.
## Running the code
To run the code, please run `main.py`. It contains a number of functions for testing the different evolutionary algorithms. Simply enter the name of the function you wish to test on line 374, along with any speficied parameters. Currently, it is set up to perform a basic run of NSGA-III, for 200 generations with a population size of 200, and display the results in a series of graphs.

