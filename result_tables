"""
AI Model Evaluation System Solver

This module solves and analyzes a system of equations representing AI model evaluations,
where different models (OpenAI, Anthropic, Google) evaluate each other and themselves.
It uses least squares optimization to find the best solution when the system is
potentially inconsistent.
"""

import numpy as np
from scipy.optimize import least_squares
import sympy
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class EvaluationSystem:
    """Represents the AI evaluation system's input data and results"""
    results_matrix: List[List[float]]
    equations: List[sympy.Eq]
    symbols: Dict[str, sympy.Symbol]

@dataclass
class Solution:
    """Contains the solved values and quality metrics"""
    chatbot_scores: List[float]  # [Anthropic, Google, OpenAI]
    judge_scores: List[float]    # [Anthropic, Google, OpenAI]
    self_bias: List[float]       # [Anthropic, Google, OpenAI]
    residuals: List[float]
    rms_error: float

def create_system() -> EvaluationSystem:
    """Create and return the evaluation system with symbols and equations"""
    # Define symbols
    symbols = {
        'OJ': sympy.Symbol('OJ', real=True),  # OpenAI Judge
        'AJ': sympy.Symbol('AJ', real=True),  # Anthropic Judge
        'GJ': sympy.Symbol('GJ', real=True),  # Google Judge
        'OC': sympy.Symbol('OC', real=True),  # OpenAI Chatbot
        'AC': sympy.Symbol('AC', real=True),  # Anthropic Chatbot
        'GC': sympy.Symbol('GC', real=True),  # Google Chatbot
        'OSB': sympy.Symbol('OSB', real=True),  # OpenAI Self Bias
        'ASB': sympy.Symbol('ASB', real=True),  # Anthropic Self Bias
        'GSB': sympy.Symbol('GSB', real=True),  # Google Self Bias
    }

    # Define equations
    equations = [
        sympy.Eq(symbols['AJ'] + symbols['OC'], 1.1),           # OpenAI-Anthropic
        sympy.Eq(symbols['GJ'] + symbols['OC'], 1.17),          # OpenAI-Google
        sympy.Eq(symbols['OJ'] + symbols['OC'] + symbols['OSB'], 1.38),   # OpenAI-OpenAI
        sympy.Eq(symbols['AJ'] + symbols['AC'] + symbols['ASB'], 1.04),   # Anthropic-Anthropic
        sympy.Eq(symbols['GJ'] + symbols['AC'], 1.15),          # Anthropic-Google
        sympy.Eq(symbols['OJ'] + symbols['AC'], 1.36),          # Anthropic-OpenAI
        sympy.Eq(symbols['AJ'] + symbols['GC'], 1.08),          # Google-Anthropic
        sympy.Eq(symbols['GJ'] + symbols['GC'] + symbols['GSB'], 1.16),   # Google-Google
        sympy.Eq(symbols['OJ'] + symbols['GC'], 1.30),          # Google-OpenAI
    ]

    # Define results matrix
    results_matrix = [
        [1.38, 1.1, 1.17],   # OpenAI evaluating [OpenAI, Anthropic, Google]
        [1.36, 1.04, 1.15],  # Anthropic evaluating [OpenAI, Anthropic, Google]
        [1.30, 1.08, 1.16],  # Google evaluating [OpenAI, Anthropic, Google]
    ]

    return EvaluationSystem(results_matrix, equations, symbols)

def objective_function(x: np.ndarray, equations: List[sympy.Eq]) -> np.ndarray:
    """Calculate residuals for the least squares optimization"""
    OJ, AJ, GJ, OC, AC, GC, OSB, ASB, GSB = x
    
    return np.array([
        AJ + OC - 1.1,           # eq1
        GJ + OC - 1.17,          # eq2
        OJ + OC + OSB - 1.38,    # eq3
        AJ + AC + ASB - 1.04,    # eq4
        GJ + AC - 1.15,          # eq5
        OJ + AC - 1.36,          # eq6
        AJ + GC - 1.08,          # eq7
        GJ + GC + GSB - 1.16,    # eq8
        OJ + GC - 1.30,          # eq9
    ])

def solve_system(system: EvaluationSystem) -> Solution:
    """Solve the system using least squares optimization and normalize results"""
    # Initial solution
    x0 = np.zeros(9)
    result = least_squares(objective_function, x0, args=(system.equations,))
    
    # Extract raw values
    solution_values = result.x
    residuals = result.fun
    rms_error = np.sqrt(np.mean(np.square(residuals)))
    
    # Extract scores
    AC, GC, OC = solution_values[4], solution_values[5], solution_values[3]  # Chatbot scores
    AJ, GJ, OJ = solution_values[1], solution_values[2], solution_values[0]  # Judge scores
    ASB, GSB, OSB = solution_values[7], solution_values[8], solution_values[6]  # Self bias
    
    # Normalize judge scores around median
    judges = [AJ, GJ, OJ]
    median_judge = sorted(judges)[1]
    
    # Adjust scores
    normalized_judges = [j - median_judge for j in judges]
    adjusted_chatbots = [AC + median_judge, GC + median_judge, OC + median_judge]
    self_biases = [ASB, GSB, OSB]
    
    return Solution(adjusted_chatbots, normalized_judges, self_biases, residuals, rms_error)

def print_results(system: EvaluationSystem, solution: Solution):
    """Print all results in a formatted way"""
    # 1. Print solution quality metrics
    print("===== Solution Quality =====")
    print(f"RMS error: {solution.rms_error:.6f}")
    print("\nEquation Residuals:")
    for i, res in enumerate(solution.residuals, 1):
        print(f"Equation {i} residual: {res:.6f}")

    # 2. Print input matrix
    print("\n===== Input Matrix =====")
    labels = ["OpenAI", "Anthropic", "Google"]
    print("           |", " | ".join(f"{c:10s}" for c in labels))
    print("-" * 50)
    for i, label in enumerate(labels):
        values = system.results_matrix[i]
        print(f"{label:10s} |", " | ".join(f"{x:10.2f}" for x in values))

    # 3. Print normalized solution
    print("\n===== Normalized Solution =====")
    print("Chatbot Scores (Anthropic, Google, OpenAI):")
    for i, score in enumerate(solution.chatbot_scores):
        print(f"  {labels[i]}: {score:.4f}")
    
    print("\nJudge Scores (normalized, median = 0):")
    for i, score in enumerate(solution.judge_scores):
        print(f"  {labels[i]}: {score:.4f}")
    
    print("\nSelf Bias:")
    for i, bias in enumerate(solution.self_bias):
        print(f"  {labels[i]}: {bias:.4f}")

def main():
    """Main function to run the analysis"""
    system = create_system()
    solution = solve_system(system)
    print_results(system, solution)

if __name__ == "__main__":
    main()
