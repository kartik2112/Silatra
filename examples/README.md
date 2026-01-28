# Examples and Tutorials

This directory contains educational examples and tutorials for learning machine learning concepts.

## Multi-Task Learning Tutorial

**File**: `multi_task_learning_tutorial.ipynb`

A comprehensive Jupyter notebook demonstrating multi-task learning with PyTorch on the MNIST dataset.

### What You'll Learn

- **Multi-Task Learning Concepts**: Understanding MTL architecture, benefits, and challenges
- **PyTorch Implementation**: Complete working example with 4 different tasks
- **Interview Preparation**: Common questions and answers about MTL
- **Best Practices**: Production-ready tips and optimization strategies

### The 4 Tasks

This tutorial implements a single neural network that learns 4 tasks simultaneously:

1. **Digit Classification** (0-9) - 10-class classification
2. **Even/Odd Prediction** - Binary classification
3. **Greater than 4** - Binary classification  
4. **Normalized Value Regression** - Continuous prediction

### Requirements

```bash
pip install torch torchvision matplotlib numpy scikit-learn
```

### Usage

```bash
jupyter notebook multi_task_learning_tutorial.ipynb
```

Or use JupyterLab, Google Colab, or any Jupyter-compatible environment.

### Key Features

✅ Complete end-to-end working implementation  
✅ Detailed explanations and comments  
✅ Interview questions with answers  
✅ Visualizations and performance metrics  
✅ Best practices for production use  
✅ Simple, educational code structure

### Topics Covered

- Hard parameter sharing architecture
- Multi-task loss functions and weighting
- Shared encoder with task-specific heads
- Training strategies for multiple tasks
- Evaluation metrics for different task types
- Common pitfalls and how to avoid them

Perfect for interview preparation, learning PyTorch, or understanding multi-task learning workflows!
