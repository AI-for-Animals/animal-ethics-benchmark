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

# Define input numbers with descriptive keys
EVALUATION_INPUTS = {
    "OpenAI judged by Anthropic": 1.87,
    "OpenAI judged by Google": 1.12,
    "OpenAI judged by OpenAI": 2.69,
    "Anthropic judged by Anthropic": 2.01,
    "Anthropic judged by Google": 1.07,
    "Anthropic judged by OpenAI": 2.72,
    "Google judged by Anthropic": 1.74,
    "Google judged by Google": 1.05,
    "Google judged by OpenAI": 2.5,
}

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

    # Define equations dynamically based on input data
    eq = EVALUATION_INPUTS
    equations = [
        sympy.Eq(symbols['AJ'] + symbols['OC'], eq["OpenAI judged by Anthropic"]),
        sympy.Eq(symbols['GJ'] + symbols['OC'], eq["OpenAI judged by Google"]),
        sympy.Eq(symbols['OJ'] + symbols['OC'] + symbols['OSB'], eq["OpenAI judged by OpenAI"]),
        sympy.Eq(symbols['AJ'] + symbols['AC'] + symbols['ASB'], eq["Anthropic judged by Anthropic"]),
        sympy.Eq(symbols['GJ'] + symbols['AC'], eq["Anthropic judged by Google"]),
        sympy.Eq(symbols['OJ'] + symbols['AC'], eq["Anthropic judged by OpenAI"]),
        sympy.Eq(symbols['AJ'] + symbols['GC'], eq["Google judged by Anthropic"]),
        sympy.Eq(symbols['GJ'] + symbols['GC'] + symbols['GSB'], eq["Google judged by Google"]),
        sympy.Eq(symbols['OJ'] + symbols['GC'], eq["Google judged by OpenAI"]),
    ]

    # Generate results matrix dynamically
    results_matrix = [
        [eq["OpenAI judged by OpenAI"], eq["OpenAI judged by Anthropic"], eq["OpenAI judged by Google"]],
        [eq["Anthropic judged by OpenAI"], eq["Anthropic judged by Anthropic"], eq["Anthropic judged by Google"]],
        [eq["Google judged by OpenAI"], eq["Google judged by Anthropic"], eq["Google judged by Google"]],
    ]

    return EvaluationSystem(results_matrix, equations, symbols)

def objective_function(x: np.ndarray, equations: List[sympy.Eq]) -> np.ndarray:
    """Calculate residuals for the least squares optimization"""
    OJ, AJ, GJ, OC, AC, GC, OSB, ASB, GSB = x

    return np.array([
        AJ + OC - EVALUATION_INPUTS["OpenAI judged by Anthropic"],           # eq1
        GJ + OC - EVALUATION_INPUTS["OpenAI judged by Google"],             # eq2
        OJ + OC + OSB - EVALUATION_INPUTS["OpenAI judged by OpenAI"],       # eq3
        AJ + AC + ASB - EVALUATION_INPUTS["Anthropic judged by Anthropic"], # eq4
        GJ + AC - EVALUATION_INPUTS["Anthropic judged by Google"],          # eq5
        OJ + AC - EVALUATION_INPUTS["Anthropic judged by OpenAI"],          # eq6
        AJ + GC - EVALUATION_INPUTS["Google judged by Anthropic"],          # eq7
        GJ + GC + GSB - EVALUATION_INPUTS["Google judged by Google"],       # eq8
        OJ + GC - EVALUATION_INPUTS["Google judged by OpenAI"],             # eq9
    ])

def solve_system(system: EvaluationSystem) -> Solution:
    """Solve the system using least squares optimization and normalize results"""
    # Initial solution
    x0 = np.zeros(9)
    result = least_squares(objective_function, x0, args=(system.equations,))

    # Extract raw values from solution
    solution_values = result.x
    residuals = result.fun
    rms_error = np.sqrt(np.mean(np.square(residuals)))

    # Ensure correct mapping
    OJ, AJ, GJ = solution_values[0], solution_values[1], solution_values[2]  # Judge scores
    OC, AC, GC = solution_values[3], solution_values[4], solution_values[5]  # Chatbot scores
    OSB, ASB, GSB = solution_values[6], solution_values[7], solution_values[8]  # Self biases

    # Normalize judge scores around median
    judges = [OJ, AJ, GJ]
    median_judge = np.median(judges)
    normalized_judges = [j - median_judge for j in judges]

    # Adjust chatbot scores for context
    adjusted_chatbots = [OC + median_judge, AC + median_judge, GC + median_judge]
    self_biases = [OSB, ASB, GSB]

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
    print("Rows represent chatbots being evaluated, columns represent judges.")
    labels = ["OpenAI", "Anthropic", "Google"]
    print(f"{'Chatbot':<12} | {'OpenAI':<12} | {'Anthropic':<12} | {'Google':<12}")
    print("-" * 58)
    for i, label in enumerate(labels):
        values = system.results_matrix[i]
        print(f"{label:<12} | " + " | ".join(f"{x:<12.2f}" for x in values))

    # 3. Print normalized solution with additional percentage column
    print("\n===== Normalized Solution =====")
    print("Rows represent chatbots and their evaluation metrics.")
    header = f"{'Chatbot':<12} | {'Chatbot Score (1-5)':<22} | {'Chatbot Score (%)':<20} | {'Judge Bias ':<15} | {'Self Bias':<12}"
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        chatbot_score = solution.chatbot_scores[i]
        chatbot_score_percent = (5 - chatbot_score) / 4 * 100
        print(
            f"{label:<12} | {chatbot_score:<22.4f} | {chatbot_score_percent:<20.2f} | {solution.judge_scores[i]:<15.4f} | {solution.self_bias[i]:<12.4f}"
        )

    print("\nNote: Positive Judge Bias indicates harsher judgments (higher scores).")



def main():
    """Main function to run the analysis"""
    system = create_system()
    solution = solve_system(system)
    print_results(system, solution)

if __name__ == "__main__":
    main()
