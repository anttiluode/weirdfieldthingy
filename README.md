# Weird Field Thingy

https://github.com/anttiluode/weirdfieldthingy/raw/main/weirdfieldthing.png


A dynamic field simulation with live webcam input built using PyTorch and Tkinter. The evolving 2D field is updated via convolution, decay, and noise, with an external bias from your webcam. You can adjust simulation parameters on-the-fly and fully reset the program to clear any repeating patterns.

## Features

- Continuous simulation with live webcam input.
- Adjustable parameters: resolution, coupling, decay, noise, input influence, and initial conditions.
- Save and load simulation states.
- Full program reset (including a new RNG seed).

## Installation

1. Clone the repository:

   git clone https://github.com/anttiluode/weirdfieldthingy.git
   cd weirdfieldthingy

Install dependencies:

pip install -r requirements.txt

# Usage

Run the application with:

python app.py

# License
MIT License
