# Metaheuristic-Practices

## Description
This repository contains a collection of Python scripts dedicated to the study, implementation, and visualization of metaheuristic optimization algorithms. The main goal of this project is to provide an interactive and visual environment to understand how different algorithms (such as Hill Climbing, Random Search, and Genetic Algorithms) explore the search space to find maximums and minimums in classic mathematical evaluation benchmark functions.

## Main Features

The project is divided into three key components, each featuring real-time 2D (contour lines) and 3D visualizations:

### 1. Objective Functions (Benchmark Functions)
The mathematical functions used to evaluate the performance of the algorithms:
* **Peaks:** A function with multiple local maximums and minimums.
* **Ackley:** A classic multimodal function, widely used to evaluate convergence.
* **Rastrigin:** A highly multimodal function, famous for its difficulty due to a large number of local minimums.
* **Sphere:** A continuous, convex, and unimodal function.

### 2. Initialization Methods
Different approaches to generating the initial population or starting points:
* **Uniform Random:** Randomly distributed points across the search space.
* **LHS (Latin Hypercube Sampling):** A stratified sampling method that ensures better coverage and representativeness of the dimensional space.
* **MaxDistance (MaxMin):** An algorithm that maximizes the minimum distance between points to guarantee maximum initial dispersion.

### 3. Optimization Algorithms
* **Local Search (Hill Climbing):** An exploitation-focused algorithm that analyzes nearby neighbors and dynamically adjusts its step size to "climb" toward the best local/global result.
* **Random Search:** A pure exploration algorithm that samples points across the entire solution space.
* **Genetic Algorithm:** A bio-inspired algorithm utilizing a population of solutions. It applies proportional roulette wheel selection, single-point crossover, and random mutation to evolve toward better solutions over generations.

## Requirements and Dependencies

To run this project, you need Python 3.x installed along with the following libraries:
- `numpy`
- `matplotlib`

You can install the required dependencies using pip:
```bash
pip install numpy matplotlib
```
# How to Use
The repository includes a main integrating script that provides an interactive menu in the console. To start evaluating the search algorithms, simply run:

```bash
python METAURISTICOS.py
```
From this interactive menu, you will be able to:
1. Select the objective function you want to optimize.
2. Choose your goal (Maximize or Minimize).
3. Select the search algorithm (Random Search or Hill Climbing).
4. (If Hill Climbing is chosen) Select the initialization method for the starting points.

Additionally, you can test the **Genetic Algorithm** directly by running its standalone script:
```bash
python FUNCIONES/GENETICO.py