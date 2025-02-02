#!/usr/bin/env python
"""
Dynamic Field with Live Webcam Input,
Complete Full Reset and Continuous Simulation with Ephaptic Coupling and Rhythmic Modulation

This application creates a dynamic 2D field that continuously evolves using a
modified convolution–decay–noise update rule that incorporates ephaptic coupling 
and intrinsic rhythmic modulation, inspired by recent research on ephaptic communication.
At each update step, an external image (from a webcam) is fed in as a bias to the field update.
The GUI (built with Tkinter) allows you to adjust:
  - Field fidelity (resolution)
  - Ephaptic strength (modified coupling)
  - Decay factor
  - Noise level
  - Input influence (the weight of the webcam image)
  - Initial condition parameters (mean and standard deviation for field reinitialization)
  - And choose the webcam index.

A single “Reset” button completely restarts the program (with a new RNG seed)
to prevent pattern repetition.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os, sys, subprocess  # For complete full reset

# --------------------- New Dynamic Field Module with Ephaptic Coupling ---------------------
class PDEField(nn.Module):
    def __init__(self, field_size=64, num_experts=4, ephaptic_range=5):
        """
        The PDEField applies several convolution "experts" to generate a local update.
        It then adds an ephaptic coupling term computed by convolving the current field
        with a distance–dependent mask. To ensure that the convolution output has the same 
        spatial dimensions as the input, we force the ephaptic mask (kernel) to have an odd size.
        """
        super().__init__()
        self.field_size = field_size
        # Ensure an odd kernel size: if field_size is even, use field_size - 1.
        self.ephaptic_kernel_size = field_size if (field_size % 2 == 1) else field_size - 1
        
        # Create several convolution experts for local updates
        self.experts = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            for _ in range(num_experts)
        ])
        # Initialize expert weights
        for expert in self.experts:
            nn.init.normal_(expert.weight, mean=0.0, std=0.1)
        
        # Create ephaptic coupling mask with size (ephaptic_kernel_size, ephaptic_kernel_size)
        distance_matrix = self._create_distance_matrix(self.ephaptic_kernel_size)
        self.ephaptic_mask = torch.exp(-distance_matrix / ephaptic_range)
        
    def _create_distance_matrix(self, size):
        # Create a matrix where each element is the Euclidean distance from the center
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        center = size // 2
        return torch.sqrt((x - center)**2 + (y - center)**2).float()
    
    def forward(self, field_state):
        # Compute outputs from each expert
        expert_outputs = []
        for expert in self.experts:
            out = expert(field_state)
            expert_outputs.append(out)
        # Average the expert outputs
        expert_mean = torch.mean(torch.stack(expert_outputs), dim=0)
        
        # Prepare the ephaptic mask kernel and compute padding
        mask = self.ephaptic_mask.unsqueeze(0).unsqueeze(0).to(field_state.device)
        # Set padding to (kernel_size - 1)//2 to preserve the spatial dimensions
        padding = (mask.shape[-1] - 1) // 2
        ephaptic = F.conv2d(field_state, mask, padding=padding)
        
        return expert_mean + 0.01 * ephaptic

class RhythmicModulator(nn.Module):
    def __init__(self, field_size, freq_bands={'theta': 6, 'alpha': 10, 'beta': 20}):
        super().__init__()
        self.field_size = field_size
        self.freq_bands = freq_bands
        # Initialize a phase for each frequency band
        self.phase = {band: 0.0 for band in freq_bands.keys()}
        
    def forward(self, field_state, dt=0.01):
        # Update phases for each frequency band
        for band, freq in self.freq_bands.items():
            self.phase[band] = (self.phase[band] + 2 * np.pi * freq * dt) % (2 * np.pi)
            
        # Create rhythmic modulation by summing sine-modulated versions of the field
        modulation = torch.zeros_like(field_state)
        for band, phase in self.phase.items():
            mod = 0.5 * (1 + torch.sin(torch.tensor(phase, device=field_state.device)))
            modulation += mod * field_state
        return modulation

class DynamicField(nn.Module):
    def __init__(self, field_size=64, decay=0.995, ephaptic_strength=0.02, noise_scale=0.02,
                 input_scale=0.2, init_mean=0.0, init_std=0.1, device='cpu'):
        """
        Initialize a dynamic field that evolves with ephaptic coupling and rhythmic modulation.
        """
        super().__init__()
        self.field_size = field_size
        self.decay = decay
        self.ephaptic_strength = ephaptic_strength
        self.noise_scale = noise_scale
        self.input_scale = input_scale
        self.init_mean = init_mean
        self.init_std = init_std
        self.device = device
        
        # Initialize the field as a 4D tensor [1, 1, field_size, field_size]
        self.field = nn.Parameter(
            torch.randn(1, 1, field_size, field_size, device=device) * self.init_std + self.init_mean,
            requires_grad=False
        )
        
        # Instantiate the PDE module with ephaptic coupling and the rhythmic modulator
        self.pde = PDEField(field_size=field_size).to(device)
        self.rhythm = RhythmicModulator(field_size=field_size).to(device)
        
    def evolve(self, external_input=None):
        """
        Update the field state with ephaptic coupling, rhythmic modulation, decay, and noise.
        Optionally, incorporate an external input (e.g., from a webcam) as bias.
        """
        # Apply rhythmic modulation to the current field
        modulated = self.rhythm(self.field)
        
        # Compute the PDE update (including ephaptic influence)
        pde_update = self.pde(modulated)
        
        # Add external input if provided (scaled by input_scale)
        if external_input is not None:
            input_tensor = torch.tensor(external_input, dtype=torch.float32, device=self.device)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Shape: [1,1,H,W]
            pde_update = pde_update + self.input_scale * input_tensor
        
        # Update the field using decay and the PDE (ephaptic) update
        new_state = self.decay * self.field + self.ephaptic_strength * pde_update
        
        # Add random noise to maintain dynamic variability
        noise = torch.randn_like(self.field) * self.noise_scale
        new_state = new_state + noise
        
        # Bound the field with a hyperbolic tangent activation
        self.field.data = torch.tanh(new_state)
        return self.field
    
    def reset_field(self):
        """Reinitialize the field using the current initial condition parameters."""
        self.field.data = torch.randn(1, 1, self.field_size, self.field_size, device=self.device) * self.init_std + self.init_mean
    
    def set_fidelity(self, new_size):
        """Adjust the resolution (fidelity) of the field via bilinear interpolation."""
        old_field = self.field
        new_field = F.interpolate(old_field, size=(new_size, new_size), mode='bilinear', align_corners=False)
        self.field = nn.Parameter(new_field, requires_grad=False)
        self.field_size = new_size
    
    def get_image(self):
        """
        Convert the current field state to an RGB image using the 'plasma' colormap.
        Returns a numpy array (uint8) of shape (field_size, field_size, 3).
        """
        field_np = self.field.squeeze().detach().cpu().numpy()
        # Normalize to [0,1]
        field_np = (field_np - field_np.min()) / (field_np.max() - field_np.min() + 1e-8)
        cmap = plt.get_cmap("plasma")
        img_rgb = (cmap(field_np)[:, :, :3] * 255).astype(np.uint8)
        return img_rgb
    
    def get_state(self):
        """Return the current state as a dictionary."""
        return {
            'field': self.field.squeeze().detach().cpu().numpy().tolist(),
            'field_size': self.field_size,
            'decay': self.decay,
            'ephaptic_strength': self.ephaptic_strength,
            'noise_scale': self.noise_scale,
            'input_scale': self.input_scale,
            'init_mean': self.init_mean,
            'init_std': self.init_std
        }
    
    def load_state(self, state):
        """Load the field state from a dictionary."""
        self.field_size = state.get('field_size', self.field_size)
        self.decay = state.get('decay', self.decay)
        self.ephaptic_strength = state.get('ephaptic_strength', self.ephaptic_strength)
        self.noise_scale = state.get('noise_scale', self.noise_scale)
        self.input_scale = state.get('input_scale', self.input_scale)
        self.init_mean = state.get('init_mean', self.init_mean)
        self.init_std = state.get('init_std', self.init_std)
        field_data = np.array(state.get('field'))
        field_tensor = torch.tensor(field_data, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        self.field.data = field_tensor

# --------------------- Tkinter GUI with Webcam Selection ---------------------
class FieldWebcamGUI:
    def __init__(self, root, simulator: DynamicField):
        self.root = root
        self.simulator = simulator
        self.root.title("Dynamic Field Webcam App with Ephaptic Coupling")
        self.status_var = tk.StringVar(value="Simulation Ready")
        self.webcam_index_var = tk.StringVar(value="0")  # Default webcam index as string
        self.cap = None
        self.running = False  # Initially, simulation is not running
        self.setup_gui()
        self.initialize_camera(int(self.webcam_index_var.get()))
        # Note: The simulation loop is started by clicking the Start Simulation button.
    
    def setup_gui(self):
        # Top control frame for simulation parameters and reset
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Reset button: completely restarts the program
        self.reset_btn = ttk.Button(control_frame, text="Reset", command=self.reset_program)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Start button: starts the simulation loop
        self.start_btn = ttk.Button(control_frame, text="Start Simulation", command=self.start_simulation_callback)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        # Fidelity slider
        ttk.Label(control_frame, text="Fidelity (Resolution):").pack(side=tk.LEFT, padx=5)
        self.fidelity_slider = ttk.Scale(control_frame, from_=16, to=128, orient=tk.HORIZONTAL, command=self.change_fidelity)
        self.fidelity_slider.set(self.simulator.field_size)
        self.fidelity_slider.pack(side=tk.LEFT, padx=5)
        
        # Ephaptic Strength slider (modified coupling)
        ttk.Label(control_frame, text="Ephaptic Strength:").pack(side=tk.LEFT, padx=5)
        self.coupling_slider = ttk.Scale(control_frame, from_=0.01, to=0.1, orient=tk.HORIZONTAL, command=self.change_coupling)
        self.coupling_slider.set(self.simulator.ephaptic_strength)
        self.coupling_slider.pack(side=tk.LEFT, padx=5)
        
        # Decay slider
        ttk.Label(control_frame, text="Decay:").pack(side=tk.LEFT, padx=5)
        self.decay_slider = ttk.Scale(control_frame, from_=0.9, to=1.0, orient=tk.HORIZONTAL, command=self.change_decay)
        self.decay_slider.set(self.simulator.decay)
        self.decay_slider.pack(side=tk.LEFT, padx=5)
        
        # Noise slider
        ttk.Label(control_frame, text="Noise:").pack(side=tk.LEFT, padx=5)
        self.noise_slider = ttk.Scale(control_frame, from_=0.0, to=0.1, orient=tk.HORIZONTAL, command=self.change_noise)
        self.noise_slider.set(self.simulator.noise_scale)
        self.noise_slider.pack(side=tk.LEFT, padx=5)
        
        # Input Influence slider
        ttk.Label(control_frame, text="Input Influence:").pack(side=tk.LEFT, padx=5)
        self.input_slider = ttk.Scale(control_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, command=self.change_input_scale)
        self.input_slider.set(self.simulator.input_scale)
        self.input_slider.pack(side=tk.LEFT, padx=5)
        
        # Frame for webcam selection
        webcam_frame = ttk.Frame(self.root)
        webcam_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        ttk.Label(webcam_frame, text="Webcam Index:").pack(side=tk.LEFT, padx=5)
        self.webcam_combo = ttk.Combobox(webcam_frame, textvariable=self.webcam_index_var, state="readonly")
        self.webcam_combo['values'] = ["0", "1", "2", "3"]
        self.webcam_combo.current(0)
        self.webcam_combo.pack(side=tk.LEFT, padx=5)
        self.set_webcam_btn = ttk.Button(webcam_frame, text="Set Webcam", command=self.change_webcam)
        self.set_webcam_btn.pack(side=tk.LEFT, padx=5)
        
        # Frame for initial field conditions
        init_frame = ttk.Frame(self.root)
        init_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        ttk.Label(init_frame, text="Initial Mean:").pack(side=tk.LEFT, padx=5)
        self.init_mean_slider = ttk.Scale(init_frame, from_=-1.0, to=1.0, orient=tk.HORIZONTAL, command=self.change_init_mean)
        self.init_mean_slider.set(self.simulator.init_mean)
        self.init_mean_slider.pack(side=tk.LEFT, padx=5)
        ttk.Label(init_frame, text="Initial Std:").pack(side=tk.LEFT, padx=5)
        self.init_std_slider = ttk.Scale(init_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL, command=self.change_init_std)
        self.init_std_slider.set(self.simulator.init_std)
        self.init_std_slider.pack(side=tk.LEFT, padx=5)
        
        # Save/Load state buttons
        state_frame = ttk.Frame(self.root)
        state_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.save_btn = ttk.Button(state_frame, text="Save State", command=self.save_state)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.load_btn = ttk.Button(state_frame, text="Load State", command=self.load_state)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # Matplotlib Figure for display
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.axis("off")
        self.image_plot = self.ax.imshow(self.simulator.get_image(), cmap="plasma")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Status label
        ttk.Label(self.root, textvariable=self.status_var).pack(side=tk.BOTTOM, pady=5)
    
    def initialize_camera(self, index):
        if self.cap is not None:
            self.cap.release()
        # Use DirectShow backend to reduce warnings on Windows.
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.simulator.field_size)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.simulator.field_size)
        self.status_var.set(f"Using webcam index {index}")
    
    def change_webcam(self):
        try:
            index = int(self.webcam_index_var.get())
            self.initialize_camera(index)
        except Exception as e:
            self.status_var.set(f"Error setting webcam: {e}")
    
    def change_fidelity(self, value):
        new_size = int(float(value))
        self.simulator.set_fidelity(new_size)
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_size)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_size)
        self.status_var.set(f"Fidelity set to {new_size}")
    
    def change_coupling(self, value):
        self.simulator.ephaptic_strength = float(value)
        self.status_var.set(f"Ephaptic Strength set to {float(value):.3f}")
    
    def change_decay(self, value):
        self.simulator.decay = float(value)
        self.status_var.set(f"Decay set to {float(value):.3f}")
    
    def change_noise(self, value):
        self.simulator.noise_scale = float(value)
        self.status_var.set(f"Noise set to {float(value):.3f}")
    
    def change_input_scale(self, value):
        self.simulator.input_scale = float(value)
        self.status_var.set(f"Input influence set to {float(value):.3f}")
    
    def change_init_mean(self, value):
        self.simulator.init_mean = float(value)
        self.status_var.set(f"Initial mean set to {float(value):.3f}")
    
    def change_init_std(self, value):
        self.simulator.init_std = float(value)
        self.status_var.set(f"Initial std set to {float(value):.3f}")
    
    def save_state(self):
        state = self.simulator.get_state()
        filename = filedialog.asksaveasfilename(defaultextension=".json", 
                                                filetypes=[("JSON Files", "*.json")])
        if filename:
            try:
                with open(filename, "w") as f:
                    json.dump(state, f)
                self.status_var.set(f"State saved to {filename}")
            except Exception as e:
                self.status_var.set(f"Error saving state: {e}")
    
    def load_state(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if filename:
            try:
                with open(filename, "r") as f:
                    state = json.load(f)
                self.simulator.load_state(state)
                self.fidelity_slider.set(self.simulator.field_size)
                self.coupling_slider.set(self.simulator.ephaptic_strength)
                self.decay_slider.set(self.simulator.decay)
                self.noise_slider.set(self.simulator.noise_scale)
                self.input_slider.set(self.simulator.input_scale)
                self.init_mean_slider.set(self.simulator.init_mean)
                self.init_std_slider.set(self.simulator.init_std)
                self.status_var.set(f"State loaded from {filename}")
            except Exception as e:
                self.status_var.set(f"Error loading state: {e}")
    
    def start_simulation_callback(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.simulation_loop, daemon=True).start()
            self.status_var.set("Simulation Running")
    
    def simulation_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (self.simulator.field_size, self.simulator.field_size))
                normalized = resized.astype(np.float32) / 255.0
            else:
                normalized = None
            self.simulator.evolve(external_input=normalized)
            img = self.simulator.get_image()
            self.image_plot.set_data(img)
            self.canvas.draw_idle()
            time.sleep(0.03)  # ~30 FPS
    
    def reset_program(self):
        """
        Perform a complete reset of the program: stop simulation, release the camera,
        destroy the current Tkinter interface, and spawn a new process.
        """
        self.status_var.set("Resetting program...")
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()
        subprocess.Popen([sys.executable] + sys.argv)
        sys.exit(0)
    
    def on_closing(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

# --------------------- Main Launcher ---------------------
def main():
    # Seed the RNG with the current time to ensure a fresh random state each start.
    torch.manual_seed(int(time.time()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulator = DynamicField(field_size=64, decay=0.995, ephaptic_strength=0.02, noise_scale=0.02,
                             input_scale=0.2, init_mean=0.0, init_std=0.1, device=device)
    root = tk.Tk()
    gui = FieldWebcamGUI(root, simulator)
    root.protocol("WM_DELETE_WINDOW", gui.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
