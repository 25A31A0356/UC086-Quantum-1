"""
Builder script to generate:
1. notebooks/Quantum_Colab.ipynb (Notebook 1: TCGA-LUAD Cancer Genomics & Classical Baselines)
2. notebooks/Run_In_The_Quantum_Computer.ipynb (Notebook 2: Qiskit Quantum Kernel Learning & IBM Quantum)
"""

import json
import os

def make_cell(cell_type, source, execution_count=None, outputs=None):
    if isinstance(source, list):
        src_lines = [l if l.endswith('\n') else l + '\n' for l in source]
        if src_lines:
            src_lines[-1] = src_lines[-1].rstrip('\n')
    else:
        src_lines = [l + '\n' for l in source.split('\n')]
        if src_lines:
            src_lines[-1] = src_lines[-1].rstrip('\n')
    
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": src_lines
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs if outputs is not None else []
    return cell

def build_notebook(cells, metadata=None):
    if metadata is None:
        metadata = {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        }
    return {
        "cells": cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5
    }

print("Notebook builder helper loaded.")
