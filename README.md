# NeuralGraphs
This project is a graph convolutional network for the MUTAG dataset, which allows for experimentation and testing of different hyperparameters, comparing their performance.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Getting Started](#getting-started)
    1. [Prerequisites](#prerequisites)
4. [Contributing](#contributing)
5. [License](#license)

## Overview

This project is made alongside a semester project report for AAU, focusing on graph convolutional networks. 
It generates a GNN model for the MUTAG dataset, while providing tools to gather insightful information, and visualizations. 
The hyperparameters can easily be adjusted for experimentation.

## Features

- Can run Bayesian optimization and grid search over a given search space
- Prints out relevant information from dataset
- Visualization of graphs from dataset
- Trains a graph covolutional network
- Measures the accuracy and loss over time
- Plots the AUC - ROC and AUC - PR curve
<!-- - Can compare differnet hyperparameters (Revisit this if unable to complete) -->

## Getting Started

To use this project navigate to the [Main](/Main/main.py) file, which contains the essentials for the project. 
After downloading it, it can be run and adjusted as desired.

### Prerequisites
To run this program it requires python, along with the following libraries.

- PyTorch
- networkX
- numpy
- pandas
- sklearn
- matplotlib
- Bayesian_Search
- csv
- seaborn

  
## Contributing
This project was made by Andreas W. Holt, Daniel H. Hansen, Frederik Melchiors, Karen M. Andersen, Mikkel D. Bjørn, and Theis R. Mathiassen. 


## License 
MIT License

Copyright (c) 2023 Theis Mathiassen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
