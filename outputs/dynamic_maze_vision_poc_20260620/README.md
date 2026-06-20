# Dynamic Maze Vision POC

This folder is a lightweight proof of concept for replacing symbolic 2D maze
observations with egocentric grayscale vision.

- renderer: `eb_jepa.datasets.dynamic_maze.vision_renderer`
- observation shape: `[4, 128, 128]`
- channel order: `up`, `down`, `left`, `right`
- implementation: analytic grid raycaster, no Chromium, no OpenGL, no Three.js
- episode seed: `7`
- rendered frames: `39`

The renderer is designed to be called at dataset time from the current dynamic
occupancy grid, agent cell, goal cell, and door cells. It can therefore render
stochastic doors on the fly without storing images for every maze state.
