import os
# Configure matplotlib cache and encourage multi-threaded BLAS BEFORE importing numpy
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANUSCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), 'Manuscript')
os.environ.setdefault('MPLCONFIGDIR', os.path.join(SCRIPT_DIR, '.mplcache'))
try:
    os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
except Exception:
    pass
# Configure BLAS/LAPACK threading for optimal multiprocessing performance
# When using multiprocessing Pool, limit BLAS threads to avoid oversubscription
# Each worker process gets 1 thread for BLAS, allowing Pool to use all cores
try:
    _nc = "1"  # Single-threaded BLAS per worker process
    for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ.setdefault(_v, _nc)
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from dataclasses import dataclass

# Optional: sympy for symbolic limits at lattice points
try:
    import sympy as sp
    _HAVE_SYMPY = True
except Exception:  # pragma: no cover
    _HAVE_SYMPY = False
import matplotlib.colors as mcolors
from multiprocessing import Pool, cpu_count

# Enable LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# Add LaTeX preamble to include amssymb package for checkmark and pifont for dingbats
plt.rc('text.latex', preamble=r'\usepackage{amssymb}\usepackage{pifont}')

# Global parameters
SMALL_REGION_EPS = 1e-12   # Treat zones with all |sigma| <= eps as zero-valued

# -------- Lightweight CPU monitor (no external deps) ---------
def _read_proc_stat():
    """Return (idle_list,total_list) from /proc/stat for each cpuN."""
    idle = []
    total = []
    try:
        with open('/proc/stat','r') as f:
            for line in f:
                if not line.startswith('cpu'):
                    break
                parts = line.split()
                if parts[0] == 'cpu':
                    # aggregate line; skip
                    continue
                vals = []
                for x in parts[1:]:
                    try:
                        vals.append(int(x))
                    except Exception:
                        vals.append(0)
                if not vals:
                    continue
                idl = vals[3] if len(vals) > 3 else 0
                tot = sum(vals)
                idle.append(idl)
                total.append(tot)
    except Exception:
        return None
    return idle, total

def _cpu_usage_per_core(prev, curr):
    if prev is None or curr is None:
        return None
    idle0, tot0 = prev
    idle1, tot1 = curr
    n = min(len(idle0), len(idle1))
    res = []
    for i in range(n):
        didle = idle1[i]-idle0[i]
        dtotal = tot1[i]-tot0[i]
        if dtotal <= 0:
            res.append(0.0)
        else:
            usage = 100.0 * (1.0 - max(0, didle)/dtotal)
            res.append(usage)
    return res

def _start_cpu_monitor(label: str = 'CPU', interval: float = 0.5):
    """Start a background thread that prints per-core CPU usage periodically.
    Returns a callable to stop the monitor.
    """
    import threading, time
    stop_evt = threading.Event()

    def run():
        prev = _read_proc_stat()
        while not stop_evt.wait(interval):
            curr = _read_proc_stat()
            usages = _cpu_usage_per_core(prev, curr)
            prev = curr
            if usages is None:
                continue
            avg = sum(usages)/len(usages) if usages else 0.0
            top = sorted(enumerate(usages), key=lambda x: x[1], reverse=True)[:8]
            top_str = ', '.join([f"cpu{idx}:{u:.0f}%" for idx,u in top])
            print(f"[{label}] avg:{avg:.0f}%  {top_str}")

    thr = threading.Thread(target=run, name='cpumon', daemon=True)
    thr.start()

    def stop():
        stop_evt.set()
        try:
            thr.join(timeout=1.0)
        except Exception:
            pass
    return stop


@dataclass
class StyleConfig:
    """Centralized matplotlib styling configuration"""
    # Figure layout
    figure_width: float = 12
    figure_height: float = 4

    # Font sizes
    title_fontsize: int = 24
    label_fontsize: int = 20
    tick_fontsize: int = 16
    rank_text_fontsize: int = 13
    colorbar_tick_fontsize: int = 12

    # Line widths and sizes
    circle_linewidth: float = 2 * 2/3
    tick_width: float = 2
    tick_length: float = 8
    tick_pad: float = 8
    colorbar_tick_length: float = 6
    colorbar_tick_width: float = 2
    spine_linewidth: float = 2

    # Circle styling
    circle_default_color: str = 'white'
    circle_anomalous_color: str = '#FF6B9D'  # Pink-red for s=3,4,5 at D=1
    circle_highlight_color: str = 'red'
    circle_edge_color: str = 'black'

    # NaN/missing data color
    nan_color: tuple = (0.85, 0.85, 0.85, 1.0)  # Light gray RGBA

    # Rank text styling
    rank_text_vertical_offset_fraction: float = -0.10  # Fraction of fontsize
    rank_text_weight: str = 'bold'

    # Title padding
    title_pad: float = 20

@dataclass
class PlotConfig:
    s_min: int = 0
    s_max: int = 7
    d_min: int = 0
    d_max: int = 7
    num_points: int = 800
    gap_units: float = 1.0
    cbar_units: float = 0.6
    left_margin: float = 0.08
    right_margin: float = 0.02
    bottom_margin: float = 0.15
    top_margin: float = 0.15
    cmaps: tuple = ("nipy_spectral", "nipy_spectral", "nipy_spectral")
    titles: tuple = (r"$\\sigma_1$", r"$\\sigma_2$", r"$\\sigma_3$")
    style: StyleConfig = None

    def __post_init__(self):
        if self.style is None:
            self.style = StyleConfig()

def alpha(k, s, d):
    """
    Compute the Alpha factor as defined in GeneralSolution.m
    Alpha[k_,s_,d_]:=(-1)^k/Product[(d+2*(s-l-2)),{l,1,k}];
    """
    if k == 0:
        return 1.0
    
    product = 1.0
    for l in range(1, k + 1):
        denominator = d + 2 * (s - l - 2)
        if abs(denominator) < 1e-10:
            return np.inf
        product *= denominator
    
    return (-1)**k / product

def mathematica_to_python(expr_str):
    """
    Convert Mathematica syntax expression to Python code.
    Handles: Pochhammer, Gamma, power notation, variable names
    """
    import re

    # Replace variable names
    expr = expr_str.replace('NInd', 'n')
    expr = expr.replace('Spin', 's')
    expr = expr.replace('Dim', 'd')

    # Replace Pochhammer[a, b] with poch(a, b) - handle nested brackets
    def replace_pochhammer(match):
        content = match.group(1)
        return f'poch({content})'

    # Replace Gamma[x] with gamma(x) - handle nested brackets
    def replace_gamma(match):
        content = match.group(1)
        return f'gamma({content})'

    # Iteratively replace Pochhammer and Gamma, handling nested cases
    while 'Pochhammer[' in expr:
        expr = re.sub(r'Pochhammer\[([^\[\]]+)\]', replace_pochhammer, expr)

    while 'Gamma[' in expr:
        expr = re.sub(r'Gamma\[([^\[\]]+)\]', replace_gamma, expr)

    # Replace power notation: handle various patterns
    # Match digit^(...) or digit^variable or digit^-digit
    expr = re.sub(r'(\d+)\^\(([^)]+)\)', r'\1**(\2)', expr)  # 2^(-1 - n)
    expr = re.sub(r'(\d+)\^([a-zA-Z_]\w*)', r'\1**\2', expr)  # 2^n
    expr = re.sub(r'(\d+)\^-(\d+)', r'\1**(-\2)', expr)  # 2^-1

    return expr

def load_lambda_coefficients_from_csv(mode="francia"):
    """
    Load lambda coefficient expressions from CSV and convert to Python functions.
    Returns a list of 6 functions: [lambda_0, lambda_1, ..., lambda_5]
    """
    import csv
    from scipy.special import gamma, poch

    csv_path = os.path.join(SCRIPT_DIR, 'GeneralSolutionCoefficients.csv')

    expressions = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                expressions.append(row[0].strip())

    # Convert each expression to a Python function
    functions = []
    for i, expr_str in enumerate(expressions):
        # Convert Mathematica syntax to Python
        py_expr = mathematica_to_python(expr_str)

        # Create a function that evaluates this expression
        def make_lambda_func(expression):
            def lambda_func(s, n, d):
                from scipy.special import gamma, poch
                try:
                    result = eval(expression)
                    return result
                except:
                    return np.nan
            return lambda_func

        functions.append(make_lambda_func(py_expr))

    return functions

# Load lambda coefficient functions from CSV for both modes
_lambda_coefficient_functions_francia = load_lambda_coefficients_from_csv(mode="francia")
_lambda_coefficient_functions_marzo = load_lambda_coefficients_from_csv(mode="marzo")

# Unsafe mode coefficient functions (Francia) - loaded from CSV
def coeff_lambda_0_unsafe(s, n, d):
    """Coefficient of lambda_0 in the vanishing expression"""
    return _lambda_coefficient_functions_francia[0](s, n, d)

def coeff_lambda_1_unsafe(s, n, d):
    """Coefficient of lambda_1 in the vanishing expression"""
    return _lambda_coefficient_functions_francia[1](s, n, d)

def coeff_lambda_2_unsafe(s, n, d):
    """Coefficient of lambda_2 in the vanishing expression"""
    return _lambda_coefficient_functions_francia[2](s, n, d)

def coeff_lambda_3_unsafe(s, n, d):
    """Coefficient of lambda_3 in the vanishing expression"""
    return _lambda_coefficient_functions_francia[3](s, n, d)

def coeff_lambda_4_unsafe(s, n, d):
    """Coefficient of lambda_4 in the vanishing expression"""
    return _lambda_coefficient_functions_francia[4](s, n, d)

def coeff_lambda_5_unsafe(s, n, d):
    """Coefficient of lambda_5 in the vanishing expression"""
    return _lambda_coefficient_functions_francia[5](s, n, d)

"""
Here are new lambda coefficient restrictions for CarloSafe mode only:
lambda_0 -> s>=1
lambda_1 -> s>=2
lambda_2 -> s>=2
lambda_3 -> s>=3
lambda_4 -> s>=0
lambda_5 -> s>=2
"""

# CarloSafe mode coefficient functions (Marzo) - loaded from CSV
def coeff_lambda_0_carlosafe(s, n, d):
    """Coefficient of lambda_0 in the vanishing expression"""
    return _lambda_coefficient_functions_marzo[0](s, n, d)

def coeff_lambda_1_carlosafe(s, n, d):
    """Coefficient of lambda_1 in the vanishing expression"""
    return _lambda_coefficient_functions_marzo[1](s, n, d)

def coeff_lambda_2_carlosafe(s, n, d):
    """Coefficient of lambda_2 in the vanishing expression"""
    return _lambda_coefficient_functions_marzo[2](s, n, d)

def coeff_lambda_3_carlosafe(s, n, d):
    """Coefficient of lambda_3 in the vanishing expression"""
    return _lambda_coefficient_functions_marzo[3](s, n, d)

def coeff_lambda_4_carlosafe(s, n, d):
    """Coefficient of lambda_4 in the vanishing expression"""
    return _lambda_coefficient_functions_marzo[4](s, n, d)

def coeff_lambda_5_carlosafe(s, n, d):
    """Coefficient of lambda_5 in the vanishing expression"""
    return _lambda_coefficient_functions_marzo[5](s, n, d)

def extract_lambda_coefficients(s, n, d, mode="francia"):
    """
    Extract coefficients of lambda_0 through lambda_5 from the vanishing expression.
    Returns a list [coeff_lambda_0, coeff_lambda_1, ..., coeff_lambda_5]
    
    Parameters:
    - s, n, d: parameters for the coefficient functions
    - mode: "francia" or "marzo" to select coefficient functions
    """
    if mode == "marzo":
        coeffs = [
            coeff_lambda_0_carlosafe(s, n, d),
            coeff_lambda_1_carlosafe(s, n, d),
            coeff_lambda_2_carlosafe(s, n, d),
            coeff_lambda_3_carlosafe(s, n, d),
            coeff_lambda_4_carlosafe(s, n, d),
            coeff_lambda_5_carlosafe(s, n, d)
        ]
    else:  # francia mode (default)
        coeffs = [
            coeff_lambda_0_unsafe(s, n, d),
            coeff_lambda_1_unsafe(s, n, d),
            coeff_lambda_2_unsafe(s, n, d),
            coeff_lambda_3_unsafe(s, n, d),
            coeff_lambda_4_unsafe(s, n, d),
            coeff_lambda_5_unsafe(s, n, d)
        ]
    
    return coeffs

def get_valid_lambda_indices(s, mode="francia"):
    """
    Determine which lambda coefficients are valid for a given value of s
    
    For Francia mode:
    lambda_0 -> s>=0
    lambda_1 -> s>=1
    lambda_2 -> s>=2
    lambda_3 -> s>=2
    lambda_4 -> s>=3
    lambda_5 -> s>=2
    
    For Marzo mode:
    lambda_0 -> s>=1
    lambda_1 -> s>=2
    lambda_2 -> s>=2
    lambda_3 -> s>=3
    lambda_4 -> s>=0
    lambda_5 -> s>=2
    
    Returns a list of indices (0-5) for valid lambda coefficients
    """
    valid_indices = []
    
    if mode == "marzo":
        # Marzo specific restrictions
        if s >= 0:
            valid_indices.append(4)  # lambda_4
        if s >= 1:
            valid_indices.append(0)  # lambda_0
        # Special case at s == 2: use λ0, λ1, λ3, λ4, λ5 (omit λ2; include λ3)
        if s == 2:
            valid_indices.extend([1, 3, 5])  # lambda_1, lambda_3, lambda_5
        elif s >= 2:
            valid_indices.extend([1, 2, 5])  # lambda_1, lambda_2, lambda_5
        if s >= 3:
            valid_indices.append(3)  # lambda_3
    else:
        # Francia mode
        if s >= 0:
            valid_indices.append(0)  # lambda_0
        if s >= 1:
            valid_indices.append(1)  # lambda_1
        if s >= 2:
            valid_indices.extend([2, 3, 5])  # lambda_2, lambda_3, lambda_5
        if s >= 3:
            valid_indices.append(4)  # lambda_4
    
    return valid_indices

def build_matrix_A(s, d, n_values=None, mode="francia"):
    """Build the coefficient matrix A for the given s, d values and list of n values"""
    if n_values is None:
        # Determine n values based on s: n = 0, 1, 2, ..., floor(s/2)
        max_n = int(np.floor(s/2))
        n_values = list(range(max_n + 1))
    
    # Determine which lambda coefficients are valid for this s and mode
    valid_lambda_indices = get_valid_lambda_indices(s, mode)
    
    num_equations = len(n_values)
    num_lambdas = len(valid_lambda_indices)
    A = np.zeros((num_equations, num_lambdas))
    
    for i, n in enumerate(n_values):
        # Get all coefficients
        all_coeffs = extract_lambda_coefficients(s, n, d, mode=mode)
        # Extract only the valid ones
        A[i, :] = [all_coeffs[j] for j in valid_lambda_indices]
    
    return A

def build_matrix_A_with_transform(s, d, n_values=None, base_mode="francia", T: np.ndarray | None = None):
    """Build coefficient matrix using a linear transform on the 6-dim lambda vector.
    - base_mode: which base formulas to use for coefficients (use 'francia').
    - T: 6x6 matrix applied to the full coefficient vector before selecting valid indices.
    """
    if n_values is None:
        max_n = int(np.floor(s/2))
        n_values = list(range(max_n + 1))

    valid_lambda_indices = get_valid_lambda_indices(s, base_mode)
    num_equations = len(n_values)
    num_lambdas = len(valid_lambda_indices)
    A = np.zeros((num_equations, num_lambdas))

    for i, n in enumerate(n_values):
        coeffs = np.array(extract_lambda_coefficients(s, n, d, mode=base_mode), dtype=float)
        if T is not None:
            coeffs = T @ coeffs
        A[i, :] = [coeffs[j] for j in valid_lambda_indices]
    return A

def compute_singular_values(A, eps=1e-15):
    """
    Compute singular values of matrix A
    
    Parameters:
    - A: input matrix
    - eps: threshold for zero singular values
    
    Returns:
    - sigma: array of singular values in descending order
    """
    # Compute singular values without equilibration
    sigma = np.linalg.svd(A, compute_uv=False)
    # Do not drop zeros: keep all singular values so zero-valued
    # singular modes are represented in the heatmap instead of
    # becoming NaN/gray via empty arrays.
    return sigma

def numerical_rank(A, tol=1e-10):
    """
    Compute the numerical rank of matrix A
    """
    sigma = np.linalg.svd(A, compute_uv=False)
    return np.sum(sigma > tol)


def compute_base_grid(cfg: PlotConfig, grid_n: int | None = None):
    n = grid_n or cfg.num_points
    s_vals = np.linspace(cfg.s_min, cfg.s_max, n)
    d_vals = np.linspace(cfg.d_min, cfg.d_max, n)
    return np.meshgrid(s_vals, d_vals)

# ---------- Symbolic helpers for lattice-only evaluation ----------

def _sym_safe_float(val):
    try:
        if val is None:
            return None
        if hasattr(val, 'is_finite') and not val.is_finite:
            return None
        v = float(val)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


@lru_cache(maxsize=None)
def _sym_alpha_numeric(k: int, s_val: int, d_symbol):
    """Sympy version of alpha(k,s,d) for a concrete integer k and s, symbolic in d.
    For k<=0 return 1.
    """
    if k is None or k <= 0:
        return sp.Integer(1)
    expr = sp.Integer(1)
    for l in range(1, k + 1):
        expr *= (d_symbol + 2 * (s_val - l - 2))
    return (-1) ** k / expr


def _sym_alpha_symbolic(k: int, s_symbol, d_symbol):
    """Sympy alpha with s symbolic (finite product over k)."""
    if k is None or k <= 0:
        return sp.Integer(1)
    expr = sp.Integer(1)
    for l in range(1, k + 1):
        expr *= (d_symbol + 2 * (s_symbol - l - 2))
    return (-1) ** k / expr


def _sym_coeff_expr(mode: str, idx: int, s_val: int, n_val: int, d):
    """Build a sympy expression for lambda-idx coefficient at fixed (s,n), symbolic in d.
    Mirrors the current numeric formulas without modifying them.
    """
    n = n_val
    s = s_val
    a = lambda kk: _sym_alpha_numeric(kk, s, d)

    if mode == "marzo":  # use *_carlosafe forms
        if idx == 0:
            return (-2*n + s)*((-2 - 2*n + s)*(-1 - 2*n + s) + ((-2*n + s)*a(n))/a(1 + n))
        if idx == 1:
            return ((-1 - 2*n + s)*(-2*n + s)*((-2 - 2*n + s)*(-4 + d - 2*n + 2*s) + ((-2*n + s)*a(n))/a(1 + n)))/2
        if idx == 2:
            return n*(-2*n + s)*((1 + 2*n - s)*(2 + 2*n - s)*(-4 + d - 2*n + 2*s) + ((2 - 2*n + s)*a(-1 + n) + (3 - d + 4*n - 3*s)*(2*n - s)*a(n))/a(1 + n))
        if idx == 3:
            return n*((-2 - 2*n + s)*(-1 - 2*n + s)*(-2*n + s) + ((2 - 2*n + s)*a(-1 + n))/a(1 + n) + ((-2*n + s)*(1 - 4*n + 2*s)*a(n))/a(1 + n))
        if idx == 4:
            return ((2*n - s)*(1 + 2*n - s)*(2 + 2*n - s)*(-a(n) - (-6 + d - 2*n + 2*s)*a(1 + n)))/(2*a(1 + n))
        if idx == 5:
            return (n*(2*n - s)*((6 - d + 2*n - 2*s)*(1 + 2*n - s)*(2 + 2*n - s) + ((-2 + 2*n - s)*a(-1 + n) + (-8*n**2 + d*(-2 + 2*n - s) + 10*n*(-1 + s) - 3*(-2 + s)*(1 + s))*a(n))/a(1 + n)))/2
    else:  # francia: use *_unsafe forms
        if idx == 0:
            return ((-2*n + s) * a(n)) / a(1 + n)
        if idx == 1:
            return (-2*n + s) * ((-2 - 2*n + s)*(-1 - 2*n + s) + ((-2*n + s)*a(n))/a(1 + n))
        if idx == 2:
            return ((-1 - 2*n + s)*(-2*n + s)*((-2 - 2*n + s)*(-4 + d - 2*n + 2*s) + ((-2*n + s)*a(n))/a(1 + n)))/2
        if idx == 3:
            return (n*((2 - 2*n + s)*a(-1 + n) - (2*n - s)*(-2 + d - 2*n + 2*s)*a(n)))/a(1 + n)
        if idx == 4:
            return n*(-2*n + s)*((1 + 2*n - s)*(2 + 2*n - s)*(-4 + d - 2*n + 2*s) + ((2 - 2*n + s)*a(-1 + n) + (3 - d + 4*n - 3*s)*(2*n - s)*a(n))/a(1 + n))
        if idx == 5:
            return n*((-2 - 2*n + s)*(-1 - 2*n + s)*(-2*n + s) + ((2 - 2*n + s)*a(-1 + n))/a(1 + n) + ((-2*n + s)*(1 - 4*n + 2*s)*a(n))/a(1 + n))
    return sp.nan


def _sym_eval_with_limit(expr, d_symbol, d_val):
    try:
        e = sp.together(expr)
        e = sp.cancel(e)
        val = e.subs(d_symbol, d_val)
        if (getattr(val, 'is_finite', True) is False) or val.has(sp.zoo) or val.has(sp.nan):
            val = sp.limit(e, d_symbol, d_val)
        return _sym_safe_float(val)
    except Exception:
        return None


def _sym_coeff_expr_with_s_symbol(mode: str, idx: int, s_symbol, n_val: int, d):
    """Build coefficient with s symbolic (uses _sym_alpha_symbolic)."""
    n = n_val
    s = s_symbol
    a = lambda kk: _sym_alpha_symbolic(kk, s, d)

    if mode == "marzo":
        if idx == 0:
            return (-2*n + s)*((-2 - 2*n + s)*(-1 - 2*n + s) + ((-2*n + s)*a(n))/a(1 + n))
        if idx == 1:
            return ((-1 - 2*n + s)*(-2*n + s)*((-2 - 2*n + s)*(-4 + d - 2*n + 2*s) + ((-2*n + s)*a(n))/a(1 + n)))/2
        if idx == 2:
            return n*(-2*n + s)*((1 + 2*n - s)*(2 + 2*n - s)*(-4 + d - 2*n + 2*s) + ((2 - 2*n + s)*a(-1 + n) + (3 - d + 4*n - 3*s)*(2*n - s)*a(n))/a(1 + n))
        if idx == 3:
            return n*((-2 - 2*n + s)*(-1 - 2*n + s)*(-2*n + s) + ((2 - 2*n + s)*a(-1 + n))/a(1 + n) + ((-2*n + s)*(1 - 4*n + 2*s)*a(n))/a(1 + n))
        if idx == 4:
            return ((2*n - s)*(1 + 2*n - s)*(2 + 2*n - s)*(-a(n) - (-6 + d - 2*n + 2*s)*a(1 + n)))/(2*a(1 + n))
        if idx == 5:
            return (n*(2*n - s)*((6 - d + 2*n - 2*s)*(1 + 2*n - s)*(2 + 2*n - s) + ((-2 + 2*n - s)*a(-1 + n) + (-8*n**2 + d*(-2 + 2*n - s) + 10*n*(-1 + s) - 3*(-2 + s)*(1 + s))*a(n))/a(1 + n)))/2
    else:
        if idx == 0:
            return ((-2*n + s) * a(n)) / a(1 + n)
        if idx == 1:
            return (-2*n + s) * ((-2 - 2*n + s)*(-1 - 2*n + s) + ((-2*n + s)*a(n))/a(1 + n))
        if idx == 2:
            return ((-1 - 2*n + s)*(-2*n + s)*((-2 - 2*n + s)*(-4 + d - 2*n + 2*s) + ((-2*n + s)*a(n))/a(1 + n)))/2
        if idx == 3:
            return (n*((2 - 2*n + s)*a(-1 + n) - (2*n - s)*(-2 + d - 2*n + 2*s)*a(n)))/a(1 + n)
        if idx == 4:
            return n*(-2*n + s)*((1 + 2*n - s)*(2 + 2*n - s)*(-4 + d - 2*n + 2*s) + ((2 - 2*n + s)*a(-1 + n) + (3 - d + 4*n - 3*s)*(2*n - s)*a(n))/a(1 + n))
        if idx == 5:
            return n*((-2 - 2*n + s)*(-1 - 2*n + s)*(-2*n + s) + ((2 - 2*n + s)*a(-1 + n))/a(1 + n) + ((-2*n + s)*(1 - 4*n + 2*s)*a(n))/a(1 + n))
    return sp.nan


def _sym_eval_limit_in_s_above(expr, s_symbol, s_val):
    try:
        e = sp.together(expr)
        e = sp.cancel(e)
        val = e.subs(s_symbol, s_val)
        if (getattr(val, 'is_finite', True) is False) or val.has(sp.zoo) or val.has(sp.nan):
            val = sp.limit(e, s_symbol, s_val, dir='+')
        return _sym_safe_float(val)
    except Exception:
        return None


def build_matrix_A_symbolic_limit(s_val, d_val, n_values=None, mode="francia"):
    if not _HAVE_SYMPY:
        return None  # signal fallback
    if n_values is None:
        max_n = int(np.floor(s_val / 2))
        n_values = list(range(max_n + 1))

    valid_lambda_indices = get_valid_lambda_indices(s_val, mode)
    d = sp.symbols('d')

    rows = []
    for n in n_values:
        row_vals = []
        ok = True
        for j in valid_lambda_indices:
            expr = _sym_coeff_expr(mode, j, int(s_val), int(n), d)
            v = _sym_eval_with_limit(expr, d, d_val)
            if v is None:
                ok = False
                break
            row_vals.append(v)
        if not ok:
            return None
        rows.append(row_vals)
    return np.array(rows, dtype=float)


def build_matrix_A_symbolic_limit_s_above(s_val, d_val, n_values=None, mode="francia"):
    """Symbolic evaluation using limit in s -> s_val from above (d fixed)."""
    if not _HAVE_SYMPY:
        return None
    if n_values is None:
        max_n = int(np.floor(s_val / 2))
        n_values = list(range(max_n + 1))

    valid_lambda_indices = get_valid_lambda_indices(s_val, mode)
    s = sp.symbols('s')
    d = sp.symbols('d')

    rows = []
    for n in n_values:
        row_vals = []
        ok = True
        for j in valid_lambda_indices:
            expr = _sym_coeff_expr_with_s_symbol(mode, j, s, int(n), d)
            # substitute d first, then take s-limit from above
            try:
                expr_d = sp.together(expr.subs(d, d_val))
                expr_d = sp.cancel(expr_d)
            except Exception:
                ok = False
                break
            v = _sym_eval_limit_in_s_above(expr_d, s, s_val)
            if v is None:
                ok = False
                break
            row_vals.append(v)
        if not ok:
            return None
        rows.append(row_vals)
    return np.array(rows, dtype=float)

def create_histogram_equalized_norm(data):
    """
    Create a histogram-equalized normalization for the given data.
    This ensures equal area for all colors in the colormap.
    
    Parameters:
    - data: 2D array of values to normalize
    
    Returns:
    - norm: matplotlib normalization object
    """
    # Extract valid (non-NaN) data
    valid_data = data[~np.isnan(data)]
    
    if len(valid_data) == 0:
        # Fallback to standard normalization if no valid data
        return mcolors.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    
    # Sort the data to compute the cumulative distribution
    sorted_data = np.sort(valid_data)
    
    # Create the cumulative distribution function (CDF)
    # This maps data values to their cumulative probability [0, 1]
    n_points = len(sorted_data)
    cdf_values = np.arange(1, n_points + 1) / n_points
    
    # Create custom normalization when data is not constant
    if len(np.unique(sorted_data)) > 1:
        # Create custom normalization class
        class HistogramEqualizedNorm(mcolors.Normalize):
            def __init__(self, data_sorted, cdf_vals):
                self.data_sorted = data_sorted
                self.cdf_vals = cdf_vals
                # Set vmin/vmax to the data range
                super().__init__(vmin=data_sorted[0], vmax=data_sorted[-1])
                
            def __call__(self, value, clip=None):
                # Handle scalar and array inputs
                value = np.asarray(value)
                
                # Find where each value falls in the sorted data
                indices = np.searchsorted(self.data_sorted, value, side='right')
                
                # Convert indices to CDF values (normalized to [0, 1])
                result = np.zeros_like(value, dtype=float)
                
                # Handle edge cases
                valid_mask = ~np.isnan(value)
                indices[indices >= len(self.cdf_vals)] = len(self.cdf_vals) - 1
                indices[indices < 0] = 0
                
                # Map to CDF values
                result[valid_mask] = self.cdf_vals[indices[valid_mask] - 1]
                result[~valid_mask] = np.nan
                
                # Ensure we're in [0, 1] range
                result = np.clip(result, 0, 1)
                
                return np.ma.array(result, mask=~valid_mask)
        
        return HistogramEqualizedNorm(sorted_data, cdf_values)
    else:
        # Fallback for constant data
        return mcolors.Normalize(vmin=sorted_data[0], vmax=sorted_data[0])

"""Define function for parallel computation of singular values"""
def compute_singular_values_at_point(args):
    """Compute singular values at a single grid point"""
    i, j, s_val, d_val, mode = args
    
    # Build the coefficient matrix (n_values determined automatically based on s)
    A = build_matrix_A(s_val, d_val, mode=mode)
    
    # Check if matrix contains infinities or NaNs
    if np.any(np.isinf(A)) or np.any(np.isnan(A)):
        return (i, j, None)
    
    # Compute singular values
    try:
        sigma = compute_singular_values(A)
        return (i, j, sigma)
    except:
        return (i, j, None)

def create_plots(mode="francia", rank_grid_override=None, highlight_differences=None):
    """Create singular value plots for given mode"""
    mode_name = mode
    filename = "Dariograph.pdf"
    
    # Load ranks from CSV file exported from Mathematica
    import csv
    rank_grid = {}
    csv_path = os.path.join(SCRIPT_DIR, 'GeneralSolutionRanks.csv')
    print(f"Loading ranks from {csv_path}...")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for d_idx, row in enumerate(reader):
            if not row:  # Skip empty rows
                continue
            for s_idx, val in enumerate(row):
                val = val.strip()
                if val == 'X':
                    # Mark as pole using special marker
                    rank_grid[(s_idx, d_idx)] = 'X'
                elif val.isdigit():
                    rank_grid[(s_idx, d_idx)] = int(val)
                else:
                    rank_grid[(s_idx, d_idx)] = None

    # Initialize arrays for the three largest singular values (default grid)
    cfg = PlotConfig()
    Sg, Dg = compute_base_grid(cfg)
    grid_n = cfg.num_points
    s_min, s_max, d_min, d_max = cfg.s_min, cfg.s_max, cfg.d_min, cfg.d_max
    singular_value_maps = [np.full((grid_n, grid_n), np.nan) for _ in range(3)]
    
    print(f"Computing singular values over continuous grid using {cpu_count()} CPU cores ({mode_name} mode)...")
    try:
        # Create list of grid points with their coordinates
        grid_points = []
        for i in range(grid_n):
            for j in range(grid_n):
                grid_points.append((i, j, Sg[j, i], Dg[j, i], mode))
        
        # Use multiprocessing pool with chunking for better load balancing
        # Chunk size chosen to balance overhead vs parallelism
        chunksize = max(1, len(grid_points) // (cpu_count() * 8))
        with Pool(cpu_count()) as pool:
            results = pool.map(compute_singular_values_at_point, grid_points, chunksize=chunksize)
        
        # Fill the singular value maps with results
        for i, j, sigma in results:
            if sigma is not None and len(sigma) > 0:
                for k in range(min(3, len(sigma))):
                    singular_value_maps[k][j, i] = sigma[k]
    except Exception as e:
        # Fallback for restricted environments: compute sequentially on a coarser grid
        print(f"Multiprocessing unavailable ({e}). Falling back to single-process coarse grid.")
        grid_n = min(200, cfg.num_points)
        Sg, Dg = compute_base_grid(cfg, grid_n)
        singular_value_maps = [np.full((grid_n, grid_n), np.nan) for _ in range(3)]
        for i in range(grid_n):
            for j in range(grid_n):
                A = build_matrix_A(Sg[j, i], Dg[j, i], mode=mode)
                if not (np.any(np.isinf(A)) or np.any(np.isnan(A))):
                    try:
                        sigma = compute_singular_values(A)
                        if sigma is not None and len(sigma) > 0:
                            for k in range(min(3, len(sigma))):
                                singular_value_maps[k][j, i] = sigma[k]
                    except Exception:
                        pass
    
    print(f"Singular value computation completed ({mode_name} mode).")
    
    # Create subplots with visually equal gaps that equal 1 x-unit.
    # We compute axes widths from the vertical extent so that
    # 1 unit along x equals 1 unit along y across all panels.
    style = cfg.style
    fig = plt.figure(figsize=(style.figure_width, style.figure_height), constrained_layout=False)

    # Margins (figure fractions)
    left_margin = cfg.left_margin
    right_margin = cfg.right_margin
    bottom_margin = cfg.bottom_margin
    top_margin = cfg.top_margin

    # Desired axes height in figure fraction
    height = 1.0 - bottom_margin - top_margin
    y_span = d_max - d_min  # 7
    x_spans = [7.0, 5.0, 3.0]
    gap_units = 1.0  # desired gap = 1 x-unit

    # Convert height fraction to width fractions using figure aspect ratio
    fig_w_in, fig_h_in = fig.get_size_inches()
    aspect_fig = fig_h_in / fig_w_in  # multiply height fraction by this to convert to width fraction per y-unit

    def widths_for_height(h_frac: float):
        unit_w_frac = aspect_fig * (h_frac / y_span)  # width fraction per x-unit
        w = [unit_w_frac * s for s in x_spans]
        gap_w = unit_w_frac * gap_units
        return w, gap_w, unit_w_frac

    w_list, gap, unit_w_frac = widths_for_height(height)
    # Colorbar width specified in x-units (slim but readable)
    cbar_units = cfg.cbar_units
    cbar_w = unit_w_frac * cbar_units

    # If total width exceeds available space, uniformly scale down height
    available_width = 1.0 - left_margin - right_margin
    # Reserve room for two internal gaps, one gap between last panel and cbar,
    # and the colorbar itself.
    total_needed = sum(w_list) + 2 * gap + gap + cbar_w
    if total_needed > available_width:
        scale = available_width / total_needed
        height *= scale
        w_list, gap, unit_w_frac = widths_for_height(height)
        cbar_w = unit_w_frac * cbar_units
    w1, w2, w3 = w_list

    # Compute left positions
    left1 = left_margin
    left2 = left1 + w1 + gap
    left3 = left2 + w2 + gap

    # Create axes at explicit positions
    ax1 = fig.add_axes([left1, bottom_margin, w1, height])
    ax2 = fig.add_axes([left2, bottom_margin, w2, height], sharey=ax1)
    ax3 = fig.add_axes([left3, bottom_margin, w3, height], sharey=ax1)
    axes = [ax1, ax2, ax3]

    # Hide duplicate y tick labels on the right-hand panels
    ax2.tick_params(labelleft=False)
    ax3.tick_params(labelleft=False)
    
    # Plot titles for the three largest singular values
    titles = [r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']
    
    # Colormaps for each subplot
    cmaps = ['nipy_spectral', 'nipy_spectral', 'nipy_spectral']  # Using nipy_spectral for all three
    
    # Create the three subplots
    for k in range(3):
        ax = axes[k]
        
        # Apply logarithmic scaling to the data
        log_data = np.log10(singular_value_maps[k] + 1e-15)  # Add small value to avoid log(0)
        
        # Define s-zone boundaries
        s_zones = [0, 1, 2, 3, 4, 6, 7]
        
        # Create a composite image where each zone is independently normalized
        composite_image = np.full_like(log_data, np.nan)
        
        # Get the colormap
        cmap = plt.get_cmap(cmaps[k])
        
        # Create RGBA image for composition
        rgba_image = np.zeros((*log_data.shape, 4))
        
        # Process each s-zone independently
        for i in range(len(s_zones) - 1):
            s_start, s_end = s_zones[i], s_zones[i + 1]
            
            # Create mask for current s-zone
            s_mask = (Sg >= s_start) & (Sg < s_end)
            
            # Extract data for this zone
            zone_data = log_data[s_mask]
            
            # Skip if zone has no valid data
            if len(zone_data) == 0 or np.all(np.isnan(zone_data)):
                continue
            
            # Decide if the zone is numerically near-zero everywhere.
            # Work in original (pre-log) sigma units for this check.
            zone_sig = np.power(10.0, zone_data) - 1e-15
            # Clip tiny negative due to roundoff
            zone_sig = np.where(np.isnan(zone_sig), np.nan, np.maximum(zone_sig, 0.0))
            valid = ~np.isnan(zone_sig)
            if np.any(valid) and np.all(np.abs(zone_sig[valid]) <= SMALL_REGION_EPS):
                # Force entire zone to map to 0 on the colormap scale
                normalized_zone = np.zeros_like(zone_data)
                zone_colors = cmap(normalized_zone)
            else:
                # Create histogram-equalized normalization for this zone
                zone_norm = create_histogram_equalized_norm(zone_data)
                # Normalize the zone data
                normalized_zone = zone_norm(log_data[s_mask])
                # Apply colormap to get RGBA values
                zone_colors = cmap(normalized_zone)
            
            # Place the colored zone data into the RGBA image
            rgba_image[s_mask] = zone_colors
        
        # Handle NaN values by making them light gray
        nan_mask = np.isnan(log_data)
        gray_color = np.array(style.nan_color)
        rgba_image[nan_mask] = gray_color
        
        # Plot the composite RGBA image
        im = ax.imshow(
            rgba_image,
            extent=[s_min, s_max, d_min, d_max],
            origin='lower',
            interpolation='bilinear',
        )
        
        # Set limits with domain restrictions and preserve scaling.
        if k == 0:  # σ₁ plot: full domain s=0-7
            ax.set_xlim(s_min, s_max)
        elif k == 1:  # σ₂ plot: restricted domain s=2-7
            ax.set_xlim(2, s_max)
        else:  # σ₃ plot: restricted domain s=4-7
            ax.set_xlim(4, s_max)
        ax.set_ylim(d_min, d_max)
        # Enforce 1:1 data scaling (1 x-unit = 1 y-unit).
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        try:
            ax.set_box_aspect((y1 - y0) / (x1 - x0))
        except Exception:
            ax.set_aspect('equal', adjustable='box')
        
        # Add numerical rank labels at integer grid points
        # First pass: Draw all circles (only within domain)
        for (s_int, d_int), rank in rank_grid.items():
            # Skip all D=0 points
            if d_int == 0:
                continue
            # Check if this point should be visible in this subplot
            if k == 1 and s_int < 2:  # σ₂: skip s<2
                continue
            elif k == 2 and s_int < 4:  # σ₃: skip s<4
                continue

            # Compute display label to determine color
            if s_int >= 3:
                display_label = 4 - rank
            elif s_int == 2:
                display_label = 3 - rank
            elif s_int == 1:
                display_label = 1 - rank
            else:  # s_int == 0
                display_label = 0 - rank

            # Set circle color based on display label value
            if display_label == 0:
                facecolor = '#66bb6a'  # bright green
            else:
                facecolor = '#dc143c'  # crimson

            # Draw circle using matplotlib Circle patch
            from matplotlib.patches import Circle
            circle = Circle((s_int, d_int), radius=0.22,
                          facecolor=facecolor, edgecolor=style.circle_edge_color,
                          linewidth=style.circle_linewidth, zorder=10, clip_on=False)
            ax.add_patch(circle)

        # Second pass: Draw text with offset (only within domain)
        for (s_int, d_int), rank in rank_grid.items():
            # Skip all D=0 points
            if d_int == 0:
                continue
            # Check if this point should be visible in this subplot
            if k == 1 and s_int < 2:  # σ₂: skip s<2
                continue
            elif k == 2 and s_int < 4:  # σ₃: skip s<4
                continue
            # Calculate a small vertical offset as a fraction of the font size
            fontsize = style.rank_text_fontsize
            # Convert font points to data coordinates for the vertical shift
            fig_height_inches = fig.get_figheight()
            axes_height_data = d_max - d_min
            points_to_data = axes_height_data / (fig_height_inches * 72)  # 72 points per inch
            vertical_offset = style.rank_text_vertical_offset_fraction * fontsize * points_to_data

            if rank == 'X':
                # Add cross mark for pole (use LaTeX bold command)
                ax.text(s_int, d_int + vertical_offset, r'\textbf{\ding{55}}',
                        ha='center', va='center',
                        fontsize=fontsize,
                        zorder=11, clip_on=False)  # Higher zorder to be on top of circle
            elif rank is not None:
                # Compute display label based on s-dependent rules
                if s_int >= 3:
                    display_label = 4 - rank
                elif s_int == 2:
                    display_label = 3 - rank
                elif s_int == 1:
                    display_label = 1 - rank
                else:  # s_int == 0
                    display_label = 0 - rank
                # Use LaTeX textbf command for bold since usetex=True
                ax.text(s_int, d_int + vertical_offset, r'\textbf{' + str(display_label) + r'}',
                        ha='center', va='center',
                        fontsize=fontsize,
                        zorder=11, clip_on=False)  # Higher zorder to be on top of circle

        # Labels and title
        ax.set_xlabel(r'$s$', fontsize=style.label_fontsize)
        if k == 0:
            ax.set_ylabel(r'$D$', fontsize=style.label_fontsize)
        ax.set_title(titles[k], fontsize=style.title_fontsize, pad=style.title_pad)

        # Set tick parameters with thicker lines
        ax.tick_params(axis='both', which='major', labelsize=style.tick_fontsize,
                      pad=style.tick_pad, width=style.tick_width, length=style.tick_length)
        
        # Set thicker spines (axes borders)
        for spine in ax.spines.values():
            spine.set_linewidth(style.spine_linewidth)
        
        # Add vertical solid lines at s = 0, 1, 2, 3, 4, 6 with 2/3 thickness of axis lines
        # Only show lines within the domain of this subplot
        for s_val in [0, 1, 2, 3, 4, 6]:
            if k == 0 and s_val <= s_max:  # σ₁: all lines
                ax.axvline(x=s_val, color='black', linestyle='-', alpha=0.8, linewidth=2*2/3)
            elif k == 1 and 2 <= s_val <= s_max:  # σ₂: s≥2
                ax.axvline(x=s_val, color='black', linestyle='-', alpha=0.8, linewidth=2*2/3)
            elif k == 2 and 4 <= s_val <= s_max:  # σ₃: s≥4
                ax.axvline(x=s_val, color='black', linestyle='-', alpha=0.8, linewidth=2*2/3)
        
        # Set integer ticks for both axes based on domain restrictions
        if k == 0:  # σ₁ plot: full domain s=0-7
            ax.set_xticks(range(int(s_min), int(s_max) + 1))
        elif k == 1:  # σ₂ plot: restricted domain s=2-7
            ax.set_xticks(range(2, int(s_max) + 1))
        else:  # σ₃ plot: restricted domain s=4-7
            ax.set_xticks(range(4, int(s_max) + 1))
        ax.set_yticks(range(int(d_min), int(d_max) + 1))

    # No colorbar needed with individual normalization.
    # GridSpec handles spacing; no need for subplots_adjust.

    # Add a standalone colorbar to the right of the rightmost axes,
    # separated by the same gap as between panels.
    try:
        import matplotlib as mpl
        cbar_left = left3 + w3 + gap
        cax = fig.add_axes([cbar_left, bottom_margin, cbar_w, height])
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0), cmap=plt.get_cmap('nipy_spectral'))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        # Minimal annotation: only 0 at bottom and 1 at top
        cb.set_ticks([0.0, 1.0])
        cb.set_ticklabels(["0", "1"])
        cb.ax.tick_params(length=style.colorbar_tick_length,
                         width=style.colorbar_tick_width,
                         labelsize=style.colorbar_tick_fontsize)
        cb.set_label("")
        # Match frame thickness to plot spines
        try:
            cb.outline.set_linewidth(style.spine_linewidth)
        except Exception:
            pass
        for sp in cax.spines.values():
            sp.set_linewidth(style.spine_linewidth)
    except Exception:
        pass
    
    # Save as PDF locally and copy to Manuscript
    import shutil
    output_path = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"PDF saved as {filename} ({mode_name} mode)")

    # Copy to Manuscript directory
    manuscript_path = os.path.join(MANUSCRIPT_DIR, filename)
    shutil.copy2(output_path, manuscript_path)
    print(f"Copied to {manuscript_path}")

def _render_transformed_frame(T: np.ndarray, rank_grid, outfile: str, grid_n: int = 200):
    """Render a single frame using transformed coefficients (base Francia)."""
    cfg = PlotConfig(num_points=grid_n)
    s_min, s_max, d_min, d_max = cfg.s_min, cfg.s_max, cfg.d_min, cfg.d_max
    Sg, Dg = compute_base_grid(cfg)
    singular_value_maps = [np.full((grid_n, grid_n), np.nan) for _ in range(3)]

    # Compute singular values (use multiprocessing when available)
    def _point_task(args):
        i, j, s_val, d_val, T_list = args
        try:
            T_mat = np.array(T_list, dtype=float)
            A = build_matrix_A_with_transform(s_val, d_val, base_mode="francia", T=T_mat)
            if np.any(np.isinf(A)) or np.any(np.isnan(A)):
                return (i, j, None)
            sigma = compute_singular_values(A)
            return (i, j, sigma)
        except Exception:
            return (i, j, None)

    points = [(i, j, Sg[j, i], Dg[j, i], T.tolist()) for i in range(grid_n) for j in range(grid_n)]
    used_pool = False
    try:
        with Pool(cpu_count()) as pool:
            for i, j, sigma in pool.map(_point_task, points):
                if sigma is not None and len(sigma) > 0:
                    for k in range(min(3, len(sigma))):
                        singular_value_maps[k][j, i] = sigma[k]
        used_pool = True
    except Exception:
        # Fallback to single process
        for args in points:
            i, j, sigma = _point_task(args)
            if sigma is not None and len(sigma) > 0:
                for k in range(min(3, len(sigma))):
                    singular_value_maps[k][j, i] = sigma[k]

    # Draw figure similar to create_plots
    fig = plt.figure(figsize=(12, 4), constrained_layout=False)
    left_margin = 0.08; right_margin = 0.02; bottom_margin = 0.15; top_margin = 0.15
    height = 1.0 - bottom_margin - top_margin
    y_span = d_max - d_min
    x_spans = [7.0, 5.0, 3.0]
    gap_units = 1.0
    fig_w_in, fig_h_in = fig.get_size_inches()
    aspect_fig = fig_h_in / fig_w_in
    def widths_for_height(h_frac: float):
        unit_w_frac = aspect_fig * (h_frac / y_span)
        w = [unit_w_frac * s for s in x_spans]
        gap_w = unit_w_frac * gap_units
        return w, gap_w, unit_w_frac
    w_list, gap, unit_w_frac = widths_for_height(height)
    cbar_units = 0.6; cbar_w = unit_w_frac * cbar_units
    available_width = 1.0 - left_margin - right_margin
    total_needed = sum(w_list) + 2 * gap + gap + cbar_w
    if total_needed > available_width:
        scale = available_width / total_needed
        height *= scale
        w_list, gap, unit_w_frac = widths_for_height(height)
        cbar_w = unit_w_frac * cbar_units
    w1, w2, w3 = w_list
    left1 = left_margin; left2 = left1 + w1 + gap; left3 = left2 + w2 + gap
    ax1 = fig.add_axes([left1, bottom_margin, w1, height])
    ax2 = fig.add_axes([left2, bottom_margin, w2, height], sharey=ax1)
    ax3 = fig.add_axes([left3, bottom_margin, w3, height], sharey=ax1)
    ax2.tick_params(labelleft=False); ax3.tick_params(labelleft=False)
    titles = [r'$\\sigma_1$', r'$\\sigma_2$', r'$\\sigma_3$']
    cmaps = ['nipy_spectral'] * 3
    for k, ax in enumerate([ax1, ax2, ax3]):
        log_data = np.log10(singular_value_maps[k] + 1e-15)
        s_zones = [0, 1, 2, 3, 4, 6, 7]
        rgba_image = np.zeros((*log_data.shape, 4))
        cmap = plt.get_cmap(cmaps[k])
        for idx in range(len(s_zones) - 1):
            s_start, s_end = s_zones[idx], s_zones[idx + 1]
            s_mask = (Sg >= s_start) & (Sg < s_end)
            zone_data = log_data[s_mask]
            if len(zone_data) == 0 or np.all(np.isnan(zone_data)):
                continue
            zone_sig = np.power(10.0, zone_data) - 1e-15
            zone_sig = np.where(np.isnan(zone_sig), np.nan, np.maximum(zone_sig, 0.0))
            valid = ~np.isnan(zone_sig)
            if np.any(valid) and np.all(np.abs(zone_sig[valid]) <= SMALL_REGION_EPS):
                normalized_zone = np.zeros_like(zone_data)
                zone_colors = cmap(normalized_zone)
            else:
                zone_norm = create_histogram_equalized_norm(zone_data)
                normalized_zone = zone_norm(log_data[s_mask])
                zone_colors = cmap(normalized_zone)
            rgba_image[s_mask] = zone_colors
        nan_mask = np.isnan(log_data)
        gray_color = np.array([0.85, 0.85, 0.85, 1.0])
        rgba_image[nan_mask] = gray_color
        ax.imshow(rgba_image, extent=[s_min, s_max, d_min, d_max], origin='lower', interpolation='bilinear')
        if k == 0:
            ax.set_xlim(s_min, s_max)
        elif k == 1:
            ax.set_xlim(2, s_max)
        else:
            ax.set_xlim(4, s_max)
        ax.set_ylim(d_min, d_max)
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        try:
            ax.set_box_aspect((y1 - y0) / (x1 - x0))
        except Exception:
            ax.set_aspect('equal', adjustable='box')
        # Draw lattice labels using Francia rank grid
        for (s_int, d_int), rank in rank_grid.items():
            if d_int == 0: continue  # Skip all D=0 points
            if k == 1 and s_int < 2: continue
            if k == 2 and s_int < 4: continue
            from matplotlib.patches import Circle

            # Compute display label to determine color
            if s_int >= 3:
                display_label = 4 - rank
            elif s_int == 2:
                display_label = 3 - rank
            elif s_int == 1:
                display_label = 1 - rank
            else:  # s_int == 0
                display_label = 0 - rank

            # Set circle color based on display label value
            if display_label == 0:
                facecolor = '#66bb6a'  # bright green
            else:
                facecolor = '#dc143c'  # crimson

            circle = Circle((s_int, d_int), radius=0.22, facecolor=facecolor, edgecolor='black', linewidth=2*2/3, zorder=10, clip_on=False)
            ax.add_patch(circle)
        for (s_int, d_int), rank in rank_grid.items():
            if d_int == 0: continue  # Skip all D=0 points
            if k == 1 and s_int < 2: continue
            if k == 2 and s_int < 4: continue
            fontsize = 13
            fig_height_inches = fig.get_figheight()
            axes_height_data = d_max - d_min
            points_to_data = axes_height_data / (fig_height_inches * 72)
            vertical_offset = -0.10 * fontsize * points_to_data
            if rank is not None:
                # Compute display label based on s-dependent rules
                if s_int >= 3:
                    display_label = 4 - rank
                elif s_int == 2:
                    display_label = 3 - rank
                elif s_int == 1:
                    display_label = 1 - rank
                else:  # s_int == 0
                    display_label = 0 - rank
                ax.text(s_int, d_int + vertical_offset, str(display_label), ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=11, clip_on=False)
            else:
                ax.text(s_int, d_int + vertical_offset, r'\\ding{55}', ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=11, clip_on=False)
        ax.set_xlabel(r'$s$', fontsize=20)
        if k == 0:
            ax.set_ylabel(r'$D$', fontsize=20)
        ax.set_title(titles[k], fontsize=24, pad=20)
        ax.tick_params(axis='both', which='major', labelsize=16, pad=8, width=2, length=8)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        for s_val in [0, 1, 2, 3, 4, 6]:
            if k == 0 and s_val <= s_max:
                ax.axvline(x=s_val, color='black', linestyle='-', alpha=0.8, linewidth=2*2/3)
            elif k == 1 and 2 <= s_val <= s_max:
                ax.axvline(x=s_val, color='black', linestyle='-', alpha=0.8, linewidth=2*2/3)
            elif k == 2 and 4 <= s_val <= s_max:
                ax.axvline(x=s_val, color='black', linestyle='-', alpha=0.8, linewidth=2*2/3)
        if k == 0:
            ax.set_xticks(range(int(s_min), int(s_max) + 1))
        elif k == 1:
            ax.set_xticks(range(2, int(s_max) + 1))
        else:
            ax.set_xticks(range(4, int(s_max) + 1))
        ax.set_yticks(range(int(d_min), int(d_max) + 1))

    # Colorbar
    try:
        import matplotlib as mpl
        cbar_left = left3 + w3 + gap
        cax = fig.add_axes([cbar_left, bottom_margin, cbar_w, height])
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0, vmax=1.0), cmap=plt.get_cmap('nipy_spectral'))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_ticks([0.0, 1.0])
        cb.set_ticklabels(["0", "1"])
        cb.ax.tick_params(length=6, width=2, labelsize=12)
        try:
            cb.outline.set_linewidth(2)
        except Exception:
            pass
        for sp in cax.spines.values():
            sp.set_linewidth(2)
    except Exception:
        pass

    os.makedirs('/home/barker/Documents/paper-v/Private/TransformFrames', exist_ok=True)
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()

def create_transform_gif(num_frames: int = 5, grid_n: int = 200, seed: int = 123):
    """Create a GIF showing evolution under random linear transforms of lambda vector.
    - Uses Francia base coefficients, preserves Francia rank labels.
    - Saves frames under Private/TransformFrames and writes Private/GeneralSolutionTransform.gif.
    """
    rank_grid = get_all_rank_grids()["francia"]
    frames_dir = '/home/barker/Documents/paper-v/Private/TransformFrames'
    os.makedirs(frames_dir, exist_ok=True)

    rng = np.random.RandomState(seed)
    frame_paths = []
    for k in range(num_frames):
        T = rng.randn(6, 6)
        # Ensure reasonable conditioning
        try:
            cond = np.linalg.cond(T)
            if not np.isfinite(cond) or cond > 1e6:
                T = T + 0.1 * np.eye(6)
        except Exception:
            T = T + 0.1 * np.eye(6)
        out_png = os.path.join(frames_dir, f'frame_{k:03d}.png')
        print(f'Rendering transformed frame {k+1}/{num_frames} -> {out_png}')
        _render_transformed_frame(T, rank_grid, out_png, grid_n=grid_n)
        frame_paths.append(out_png)

    # Try to assemble GIF using Pillow
    gif_path = '/home/barker/Documents/paper-v/Private/GeneralSolutionTransform.gif'
    try:
        from PIL import Image
        images = [Image.open(p) for p in frame_paths]
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=600, loop=0)
        print(f'GIF saved as {gif_path}')
    except Exception as e:
        print(f'PIL unavailable or failed ({e}). GIF not created; frames are available in {frames_dir}')

def create_transform_gif_fast(num_frames: int = 50, grid_n: int = 8, seed: int = 123):
    """Fast path GIF generation within tight time budget.
    - Uses small grid and batched SVD per s-band.
    - Minimal annotations; LaTeX disabled for speed.
    - Saves frames under Private/TransformFramesFast and assembles GIF.
    """
    # Prepare grid
    cfg = PlotConfig(num_points=grid_n)
    s_min, s_max, d_min, d_max = cfg.s_min, cfg.s_max, cfg.d_min, cfg.d_max
    Sg, Dg = compute_base_grid(cfg)

    # Encourage threaded BLAS usage for SVD
    try:
        import os as _os
        from multiprocessing import cpu_count as _cc
        _n = str(max(1, _cc()))
        for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
            if _os.environ.get(_v) is None:
                _os.environ[_v] = _n
    except Exception:
        pass

    # Precompute base coefficients per grid point and per s-band
    bands = []  # list of dict(mask, rows, valid_idx, C, idx_list)
    # Define s-bands for Francia: determines rows and valid columns
    band_defs = [
        {"s0": 0.0, "s1": 1.0, "rows": 1, "valid": [0]},
        {"s0": 1.0, "s1": 2.0, "rows": 1, "valid": [0,1]},
        {"s0": 2.0, "s1": 3.0, "rows": 2, "valid": [0,1,2,3,5]},
        {"s0": 3.0, "s1": 7.0, "rows": None, "valid": [0,1,2,3,4,5]},
    ]
    # Build masks and base coefficient tensors
    for bd in band_defs:
        s0, s1 = bd["s0"], bd["s1"]
        mask = (Sg >= s0) & (Sg < s1)
        idxs = np.argwhere(mask)
        if idxs.size == 0:
            bands.append({"mask": mask, "idxs": idxs, "rows": 0, "valid": bd["valid"], "C": None})
            continue
        # Determine rows by s; for last band rows vary: rows = floor(s/2)+1
        rows_list = []
        Cs = []
        for (j,i) in idxs:
            s = float(Sg[j,i]); d = float(Dg[j,i])
            r = int(np.floor(s/2)) + 1
            if bd["rows"] is not None:
                r = bd["rows"]
            # Build c for n=0..r-1
            c_rows = []
            for n in range(r):
                coeffs = extract_lambda_coefficients(s, n, d, mode="francia")
                c_rows.append(coeffs)
            Cs.append(np.array(c_rows, dtype=float))  # shape (r,6)
            rows_list.append(r)
        # For batched SVD, we need uniform rows per band; ensure last band uses max rows=4
        max_r = max(rows_list) if rows_list else 0
        C_pad = []
        for C in Cs:
            if C.shape[0] < max_r:
                pad = np.zeros((max_r - C.shape[0], 6))
                C = np.vstack([C, pad])
            C_pad.append(C)
        C_arr = np.stack(C_pad, axis=0) if C_pad else None  # (P, max_r, 6)
        bands.append({"mask": mask, "idxs": idxs, "rows": (bd["rows"] or max_r), "valid": bd["valid"], "C": C_arr})

    # Prepare output
    frames_dir = '/home/barker/Documents/paper-v/Private/TransformFramesFast'
    os.makedirs(frames_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    frame_paths = []

    # Temporarily disable LaTeX for speed
    prev_usetex = plt.rcParams.get('text.usetex', False)
    plt.rcParams['text.usetex'] = False

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _svd_for_band(bd, T):
            idxs = bd["idxs"]
            C = bd["C"]
            if C is None or idxs.size == 0:
                return (idxs, None)
            Ct = C @ T.T
            V = bd["valid"]
            A = Ct[:, :, V]
            finite_mask = np.isfinite(A).all(axis=(1,2))
            svals = np.zeros((A.shape[0], min(A.shape[1], A.shape[2])))
            good = np.where(finite_mask)[0]
            if good.size:
                try:
                    svals_good = np.linalg.svd(A[good], compute_uv=False)
                except Exception:
                    svals_good = np.array([np.linalg.svd(a, compute_uv=False) for a in A[good]])
                svals[good, :svals_good.shape[1]] = svals_good
            return (idxs, svals)

        for k in range(num_frames):
            T = rng.randn(6,6)
            # Slightly regularize
            T += 0.05 * np.eye(6)

            # Initialize sigma maps
            sv_maps = [np.full(Sg.shape, np.nan) for _ in range(3)]

            # Process each band in parallel threads
            futures = []
            with ThreadPoolExecutor(max_workers=(os.cpu_count() or 4)) as ex:
                for bd in bands:
                    futures.append(ex.submit(_svd_for_band, bd, T))
                for fut in as_completed(futures):
                    idxs, svals = fut.result()
                    if svals is None:
                        continue
                    P = svals.shape[0]
                    m = svals.shape[1]
                    s_pad = np.zeros((P, 3))
                    s_pad[:, :min(3, m)] = svals[:, :min(3, m)]
                    for p, (j,i) in enumerate(idxs):
                        sv_maps[0][j,i] = s_pad[p,0]
                        sv_maps[1][j,i] = s_pad[p,1]
                        sv_maps[2][j,i] = s_pad[p,2]

            # Render minimal figure (no labels for speed)
            fig = plt.figure(figsize=(9, 3), constrained_layout=False)
            left_margin = 0.06; right_margin = 0.01; bottom_margin = 0.12; top_margin = 0.12
            height = 1.0 - bottom_margin - top_margin
            y_span = d_max - d_min
            x_spans = [7.0, 5.0, 3.0]
            fig_w_in, fig_h_in = fig.get_size_inches(); aspect_fig = fig_h_in / fig_w_in
            def widths_for_height(h_frac):
                unit_w_frac = aspect_fig * (h_frac / y_span)
                w = [unit_w_frac * s for s in x_spans]
                gap_w = unit_w_frac * 1.0
                return w, gap_w
            w_list, gap = widths_for_height(height)
            w1,w2,w3 = w_list
            left1 = left_margin; left2 = left1 + w1 + gap; left3 = left2 + w2 + gap
            ax1 = fig.add_axes([left1, bottom_margin, w1, height])
            ax2 = fig.add_axes([left2, bottom_margin, w2, height], sharey=ax1)
            ax3 = fig.add_axes([left3, bottom_margin, w3, height], sharey=ax1)
            for kpanel, (ax, data) in enumerate(zip([ax1,ax2,ax3], sv_maps)):
                log_data = np.log10(data + 1e-15)
                rgba = np.zeros((*log_data.shape, 4))
                s_zones = [0,1,2,3,4,6,7]
                cmap = plt.get_cmap('nipy_spectral')
                for zi in range(len(s_zones)-1):
                    s0,s1 = s_zones[zi], s_zones[zi+1]
                    mask = (Sg >= s0) & (Sg < s1)
                    zd = log_data[mask]
                    if zd.size == 0 or np.all(np.isnan(zd)): continue
                    zsig = np.power(10.0, zd) - 1e-15
                    zsig = np.where(np.isnan(zsig), np.nan, np.maximum(zsig, 0.0))
                    valid = ~np.isnan(zsig)
                    if np.any(valid) and np.all(np.abs(zsig[valid]) <= SMALL_REGION_EPS):
                        normed = np.zeros_like(zd)
                    else:
                        norm = create_histogram_equalized_norm(zd)
                        normed = norm(log_data[mask])
                    rgba[mask] = cmap(normed)
                nan_mask = np.isnan(log_data)
                rgba[nan_mask] = np.array([0.85,0.85,0.85,1.0])
                ax.imshow(rgba, extent=[s_min,s_max,d_min,d_max], origin='lower', interpolation='nearest')
                if kpanel == 0: ax.set_xlim(s_min,s_max)
                elif kpanel == 1: ax.set_xlim(2,s_max)
                else: ax.set_xlim(4,s_max)
                ax.set_ylim(d_min,d_max)
                ax.set_xticks([]); ax.set_yticks([])
            out_png = os.path.join(frames_dir, f'frame_{k:03d}.png')
            plt.savefig(out_png, dpi=100, bbox_inches='tight')
            plt.close(fig)
            frame_paths.append(out_png)

        # Assemble GIF
        gif_path = '/home/barker/Documents/paper-v/Private/GeneralSolutionTransform.gif'
        try:
            from PIL import Image
            images = [Image.open(p) for p in frame_paths]
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=80, loop=0)
            print(f'GIF (fast) saved as {gif_path}')
        except Exception as e:
            print(f'PIL unavailable or failed ({e}). GIF not created; frames in {frames_dir}')
    finally:
        plt.rcParams['text.usetex'] = prev_usetex

def create_transform_gif_smooth(num_frames: int = 10, grid_n: int = 50, seed: int = 123, monitor_cpu: bool = False):
    """Create a GIF where each frame uses a transform matrix interpolated
    smoothly between two random endpoints T0 -> T1.

    - Uses Francia base coefficients.
    - Parallel band SVD similar to create_transform_gif_fast.
    - Saves frames to Private/TransformFramesFast and writes
      Private/GeneralSolutionTransform.gif
    """
    # Encourage threaded BLAS usage for SVD
    try:
        import os as _os
        from multiprocessing import cpu_count as _cc
        _n = str(max(1, _cc()))
        for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
            if _os.environ.get(_v) is None:
                _os.environ[_v] = _n
    except Exception:
        pass

    # Prepare grid
    cfg = PlotConfig(num_points=grid_n)
    s_min, s_max, d_min, d_max = cfg.s_min, cfg.s_max, cfg.d_min, cfg.d_max
    Sg, Dg = compute_base_grid(cfg)

    # Precompute base coefficients per grid point and per s-band
    bands = []
    band_defs = [
        {"s0": 0.0, "s1": 1.0, "rows": 1, "valid": [0]},
        {"s0": 1.0, "s1": 2.0, "rows": 1, "valid": [0,1]},
        {"s0": 2.0, "s1": 3.0, "rows": 2, "valid": [0,1,2,3,5]},
        {"s0": 3.0, "s1": 7.0, "rows": None, "valid": [0,1,2,3,4,5]},
    ]
    for bd in band_defs:
        s0, s1 = bd["s0"], bd["s1"]
        mask = (Sg >= s0) & (Sg < s1)
        idxs = np.argwhere(mask)
        if idxs.size == 0:
            bands.append({"mask": mask, "idxs": idxs, "rows": 0, "valid": bd["valid"], "C": None})
            continue
        rows_list = []
        Cs = []
        for (j,i) in idxs:
            s = float(Sg[j,i]); d = float(Dg[j,i])
            r = int(np.floor(s/2)) + 1
            if bd["rows"] is not None:
                r = bd["rows"]
            c_rows = []
            for n in range(r):
                coeffs = extract_lambda_coefficients(s, n, d, mode="francia")
                c_rows.append(coeffs)
            Cs.append(np.array(c_rows, dtype=float))
            rows_list.append(r)
        max_r = max(rows_list) if rows_list else 0
        C_pad = []
        for C in Cs:
            if C.shape[0] < max_r:
                pad = np.zeros((max_r - C.shape[0], 6))
                C = np.vstack([C, pad])
            C_pad.append(C)
        C_arr = np.stack(C_pad, axis=0) if C_pad else None
        bands.append({"mask": mask, "idxs": idxs, "rows": (bd["rows"] or max_r), "valid": bd["valid"], "C": C_arr})

    frames_dir = '/home/barker/Documents/paper-v/Private/TransformFramesFast'
    os.makedirs(frames_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    # Endpoints and interpolation schedule
    T0 = rng.randn(6,6) + 0.05*np.eye(6)
    T1 = rng.randn(6,6) + 0.05*np.eye(6)

    # Temporarily disable LaTeX for speed
    prev_usetex = plt.rcParams.get('text.usetex', False)
    plt.rcParams['text.usetex'] = False

    from concurrent.futures import ThreadPoolExecutor, as_completed
    frame_paths = []
    try:
        stop_mon = None
        if monitor_cpu:
            stop_mon = _start_cpu_monitor('GIF', interval=0.5)
        def _svd_for_band(bd, T):
            idxs = bd["idxs"]
            C = bd["C"]
            if C is None or idxs.size == 0:
                return (idxs, None)
            Ct = C @ T.T
            V = bd["valid"]
            A = Ct[:, :, V]
            finite_mask = np.isfinite(A).all(axis=(1,2))
            svals = np.zeros((A.shape[0], min(A.shape[1], A.shape[2])))
            good = np.where(finite_mask)[0]
            if good.size:
                try:
                    svals_good = np.linalg.svd(A[good], compute_uv=False)
                except Exception:
                    svals_good = np.array([np.linalg.svd(a, compute_uv=False) for a in A[good]])
                svals[good, :svals_good.shape[1]] = svals_good
            return (idxs, svals)

        for k in range(num_frames):
            t = 0.0 if num_frames<=1 else k/(num_frames-1)
            T = (1.0 - t)*T0 + t*T1
            sv_maps = [np.full(Sg.shape, np.nan) for _ in range(3)]
            with ThreadPoolExecutor(max_workers=(os.cpu_count() or 4)) as ex:
                futures = [ex.submit(_svd_for_band, bd, T) for bd in bands]
                for fut in as_completed(futures):
                    idxs, svals = fut.result()
                    if svals is None:
                        continue
                    P = svals.shape[0]
                    m = svals.shape[1]
                    s_pad = np.zeros((P, 3))
                    s_pad[:, :min(3, m)] = svals[:, :min(3, m)]
                    for p, (j,i) in enumerate(idxs):
                        sv_maps[0][j,i] = s_pad[p,0]
                        sv_maps[1][j,i] = s_pad[p,1]
                        sv_maps[2][j,i] = s_pad[p,2]

            # Render minimal figure (no labels for speed)
            fig = plt.figure(figsize=(9, 3), constrained_layout=False)
            left_margin = 0.06; right_margin = 0.01; bottom_margin = 0.12; top_margin = 0.12
            height = 1.0 - bottom_margin - top_margin
            y_span = d_max - d_min
            x_spans = [7.0, 5.0, 3.0]
            fig_w_in, fig_h_in = fig.get_size_inches(); aspect_fig = fig_h_in / fig_w_in
            def widths_for_height(h_frac):
                unit_w_frac = aspect_fig * (h_frac / y_span)
                w = [unit_w_frac * s for s in x_spans]
                gap_w = unit_w_frac * 1.0
                return w, gap_w
            w_list, gap = widths_for_height(height)
            w1,w2,w3 = w_list
            left1 = left_margin; left2 = left1 + w1 + gap; left3 = left2 + w2 + gap
            ax1 = fig.add_axes([left1, bottom_margin, w1, height])
            ax2 = fig.add_axes([left2, bottom_margin, w2, height], sharey=ax1)
            ax3 = fig.add_axes([left3, bottom_margin, w3, height], sharey=ax1)
            for kpanel, (ax, data) in enumerate(zip([ax1,ax2,ax3], sv_maps)):
                log_data = np.log10(data + 1e-15)
                rgba = np.zeros((*log_data.shape, 4))
                s_zones = [0,1,2,3,4,6,7]
                cmap = plt.get_cmap('nipy_spectral')
                for zi in range(len(s_zones)-1):
                    s0,s1 = s_zones[zi], s_zones[zi+1]
                    mask = (Sg >= s0) & (Sg < s1)
                    zd = log_data[mask]
                    if zd.size == 0 or np.all(np.isnan(zd)): continue
                    zsig = np.power(10.0, zd) - 1e-15
                    zsig = np.where(np.isnan(zsig), np.nan, np.maximum(zsig, 0.0))
                    valid = ~np.isnan(zsig)
                    if np.any(valid) and np.all(np.abs(zsig[valid]) <= SMALL_REGION_EPS):
                        normed = np.zeros_like(zd)
                    else:
                        norm = create_histogram_equalized_norm(zd)
                        normed = norm(log_data[mask])
                    rgba[mask] = cmap(normed)
                nan_mask = np.isnan(log_data)
                rgba[nan_mask] = np.array([0.85,0.85,0.85,1.0])
                ax.imshow(rgba, extent=[s_min,s_max,d_min,d_max], origin='lower', interpolation='nearest')
                if kpanel == 0: ax.set_xlim(s_min,s_max)
                elif kpanel == 1: ax.set_xlim(2,s_max)
                else: ax.set_xlim(4,s_max)
                ax.set_ylim(d_min,d_max)
                ax.set_xticks([]); ax.set_yticks([])
            out_png = os.path.join(frames_dir, f'frame_{k:03d}.png')
            plt.savefig(out_png, dpi=100, bbox_inches='tight')
            plt.close(fig)
            frame_paths.append(out_png)

        # Assemble GIF
        gif_path = '/home/barker/Documents/paper-v/Private/GeneralSolutionTransform.gif'
        try:
            from PIL import Image
            images = [Image.open(p) for p in frame_paths]
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=80, loop=0)
            print(f'GIF (smooth) saved as {gif_path}')
        except Exception as e:
            print(f'PIL unavailable or failed ({e}). GIF not created; frames in {frames_dir}')
    finally:
        plt.rcParams['text.usetex'] = prev_usetex
        try:
            if monitor_cpu and stop_mon:
                stop_mon()
        except Exception:
            pass

def _compute_svd_for_gif(args):
    """Helper function for parallel GIF frame computation"""
    j, i, s_val, d_val, T_list = args
    T_arr = np.array(T_list)
    try:
        A = build_matrix_A_with_transform(s_val, d_val, base_mode="francia", T=T_arr)
        if A is not None and not (np.any(np.isinf(A)) or np.any(np.isnan(A))):
            sigma = compute_singular_values(A)
            if sigma is not None and len(sigma) > 0:
                return (j, i, sigma[:min(3, len(sigma))])
    except Exception:
        pass
    return (j, i, None)

def create_transform_gif_smooth_looped(num_frames: int = 50, grid_n: int = 835, seed: int = 9, segments: int = 5, monitor_cpu: bool = False):
    """Smooth interpolation across multiple random transforms and loop back.
    - Generates `segments` random 6x6 matrices T0..T{K-1}, and interpolates
      T0->T1->...->T{K-1}->T0 across the total frames for a seamless loop.
    - Uses Francia base coefficients and the same fast, parallel banded SVD pipeline.
    - Saves frames to Private/TransformFramesFast and assembles Private/GeneralSolutionTransform.gif
    """
    # Encourage threaded BLAS usage for SVD
    try:
        import os as _os
        from multiprocessing import cpu_count as _cc
        _n = str(max(1, _cc()))
        for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
            if _os.environ.get(_v) is None:
                _os.environ[_v] = _n
    except Exception:
        pass

    # Prepare grid
    cfg = PlotConfig(num_points=grid_n)
    s_min, s_max, d_min, d_max = cfg.s_min, cfg.s_max, cfg.d_min, cfg.d_max
    Sg, Dg = compute_base_grid(cfg)

    # Load ranks from CSV
    import csv
    rank_grid = {}
    csv_path = os.path.join(SCRIPT_DIR, 'GeneralSolutionRanks.csv')
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for d_idx, row in enumerate(reader):
            if not row:
                continue
            for s_idx, val in enumerate(row):
                val = val.strip()
                if val == 'X':
                    rank_grid[(s_idx, d_idx)] = 'X'
                elif val.isdigit():
                    rank_grid[(s_idx, d_idx)] = int(val)
                else:
                    rank_grid[(s_idx, d_idx)] = None

    frames_dir = '/home/barker/Documents/paper-v/Private/TransformFramesFast'
    os.makedirs(frames_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    # Endpoint sequence and segment schedule
    K = max(2, int(segments))
    # Use truly random matrices with order unity elements
    Ts = [rng.randn(6,6) for _ in range(K)]
    frames_per_seg = max(1, num_frames // K)

    # Keep LaTeX enabled to match Francia PDF formatting
    prev_usetex = plt.rcParams.get('text.usetex', False)
    # Don't disable LaTeX - keep it enabled for consistent formatting

    frame_paths = []
    stop_mon = None
    try:
        if monitor_cpu:
            stop_mon = _start_cpu_monitor('GIF', interval=0.5)

        # Create Pool ONCE and reuse across all frames for max efficiency
        with Pool(cpu_count()) as pool:
            f_idx = 0
            for sidx in range(K):
                T0 = Ts[sidx]
                T1 = Ts[(sidx+1) % K]
                for m in range(frames_per_seg):
                    if f_idx >= num_frames:
                        break
                    t = 0.0 if frames_per_seg<=1 else m/(frames_per_seg-1)
                    T = (1.0 - t)*T0 + t*T1

                    # Compute SVD for each grid point using parallel processing
                    sv_maps = [np.full(Sg.shape, np.nan) for _ in range(3)]

                    grid_points = [(j, i, float(Sg[j, i]), float(Dg[j, i]), T.tolist())
                                  for j in range(grid_n) for i in range(grid_n)]

                    # Use all CPU cores with chunking
                    chunksize = max(1, len(grid_points) // (cpu_count() * 4))
                    results = pool.map(_compute_svd_for_gif, grid_points, chunksize=chunksize)

                    # Fill in results
                    for j, i, sigma in results:
                        if sigma is not None:
                            for k in range(len(sigma)):
                                sv_maps[k][j, i] = sigma[k]

                    # Render figure with full Francia formatting
                    fig = plt.figure(figsize=(12, 4), constrained_layout=False)
                    left_margin = 0.08; right_margin = 0.02; bottom_margin = 0.15; top_margin = 0.15
                    height = 1.0 - bottom_margin - top_margin
                    y_span = d_max - d_min
                    x_spans = [7.0, 5.0, 3.0]
                    gap_units = 1.0
                    fig_w_in, fig_h_in = fig.get_size_inches(); aspect_fig = fig_h_in / fig_w_in
                    def widths_for_height(h_frac):
                        unit_w_frac = aspect_fig * (h_frac / y_span)
                        w = [unit_w_frac * s for s in x_spans]
                        gap_w = unit_w_frac * gap_units
                        return w, gap_w, unit_w_frac
                    w_list, gap, unit_w_frac = widths_for_height(height)
                    cbar_units = 0.6
                    cbar_w = unit_w_frac * cbar_units
                    available_width = 1.0 - left_margin - right_margin
                    total_needed = sum(w_list) + 2 * gap + gap + cbar_w
                    if total_needed > available_width:
                        scale = available_width / total_needed
                        height *= scale
                        w_list, gap, unit_w_frac = widths_for_height(height)
                        cbar_w = unit_w_frac * cbar_units
                    w1, w2, w3 = w_list
                    left1 = left_margin
                    left2 = left1 + w1 + gap
                    left3 = left2 + w2 + gap
                    ax1 = fig.add_axes([left1, bottom_margin, w1, height])
                    ax2 = fig.add_axes([left2, bottom_margin, w2, height], sharey=ax1)
                    ax3 = fig.add_axes([left3, bottom_margin, w3, height], sharey=ax1)
                    axes = [ax1, ax2, ax3]
                    ax2.tick_params(labelleft=False)
                    ax3.tick_params(labelleft=False)
                    titles = [r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']
                    cmaps = ['nipy_spectral', 'nipy_spectral', 'nipy_spectral']

                    for k in range(3):
                        ax = axes[k]
                        log_data = np.log10(sv_maps[k] + 1e-15)
                        s_zones = [0, 1, 2, 3, 4, 6, 7]
                        composite_image = np.full_like(log_data, np.nan)
                        cmap = plt.get_cmap(cmaps[k])
                        rgba_image = np.zeros((*log_data.shape, 4))

                        for i in range(len(s_zones) - 1):
                            s_start, s_end = s_zones[i], s_zones[i + 1]
                            s_mask = (Sg >= s_start) & (Sg < s_end)
                            zone_data = log_data[s_mask]
                            if len(zone_data) == 0 or np.all(np.isnan(zone_data)):
                                continue
                            zone_sig = np.power(10.0, zone_data) - 1e-15
                            zone_sig = np.where(np.isnan(zone_sig), np.nan, np.maximum(zone_sig, 0.0))
                            valid = ~np.isnan(zone_sig)
                            if np.any(valid) and np.all(np.abs(zone_sig[valid]) <= SMALL_REGION_EPS):
                                normalized_zone = np.zeros_like(zone_data)
                                zone_colors = cmap(normalized_zone)
                            else:
                                zone_norm = create_histogram_equalized_norm(zone_data)
                                normalized_zone = zone_norm(log_data[s_mask])
                                zone_colors = cmap(normalized_zone)
                            rgba_image[s_mask] = zone_colors

                        nan_mask = np.isnan(log_data)
                        gray_color = np.array([0.85, 0.85, 0.85, 1.0])
                        rgba_image[nan_mask] = gray_color

                        im = ax.imshow(rgba_image, extent=[s_min, s_max, d_min, d_max],
                                     origin='lower', interpolation='bilinear')

                        if k == 0:
                            ax.set_xlim(s_min, s_max)
                        elif k == 1:
                            ax.set_xlim(2, s_max)
                        else:
                            ax.set_xlim(4, s_max)
                        ax.set_ylim(d_min, d_max)

                        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
                        try:
                            ax.set_box_aspect((y1 - y0) / (x1 - x0))
                        except Exception:
                            ax.set_aspect('equal', adjustable='box')

                        ax.set_xlabel(r'$s$', fontsize=12)
                        if k == 0:
                            ax.set_ylabel(r'$D$', fontsize=12)
                        ax.set_title(titles[k], fontsize=14, pad=10)

                        x_ticks = list(range(int(np.ceil(x0)), int(np.floor(x1)) + 1))
                        y_ticks = list(range(int(np.ceil(y0)), int(np.floor(y1)) + 1))
                        ax.set_xticks(x_ticks)
                        ax.set_yticks(y_ticks)
                        ax.tick_params(axis='both', which='major', labelsize=10, width=2, length=8, pad=8)

                        # Set thicker spines (axes borders)
                        for spine in ax.spines.values():
                            spine.set_linewidth(2)

                        # Add vertical solid lines at s = 0, 1, 2, 3, 4, 6
                        for s_val in [0, 1, 2, 3, 4, 6]:
                            if k == 0 and s_val <= s_max:  # σ₁: all lines
                                ax.axvline(x=s_val, color='black', linestyle='-', alpha=0.8, linewidth=2*2/3)
                            elif k == 1 and 2 <= s_val <= s_max:  # σ₂: s≥2
                                ax.axvline(x=s_val, color='black', linestyle='-', alpha=0.8, linewidth=2*2/3)
                            elif k == 2 and 4 <= s_val <= s_max:  # σ₃: s≥4
                                ax.axvline(x=s_val, color='black', linestyle='-', alpha=0.8, linewidth=2*2/3)

                        # Add rank labels with circles
                        from matplotlib.patches import Circle
                        # First pass: draw circles
                        for (s_int, d_int), rank in rank_grid.items():
                            if d_int == 0:  # Skip all D=0 points
                                continue
                            if k == 1 and s_int < 2:
                                continue
                            elif k == 2 and s_int < 4:
                                continue

                            # Compute display label to determine color
                            if s_int >= 3:
                                display_label = 4 - rank
                            elif s_int == 2:
                                display_label = 3 - rank
                            elif s_int == 1:
                                display_label = 1 - rank
                            else:  # s_int == 0
                                display_label = 0 - rank

                            # Set circle color based on display label value
                            if display_label == 0:
                                facecolor = '#c8e6c9'  # pale green
                            elif display_label == 1:
                                facecolor = '#ffcdd2'  # light red
                            elif display_label == 2:
                                facecolor = '#ef5350'  # medium red
                            elif display_label == 3:
                                facecolor = '#c62828'  # intense red
                            else:
                                facecolor = 'white'  # default for any other values

                            circle = Circle((s_int, d_int), radius=0.22,
                                          facecolor=facecolor, edgecolor='black',
                                          linewidth=2*2/3, zorder=10, clip_on=False)
                            ax.add_patch(circle)

                        # Second pass: draw text
                        for (s_int, d_int), rank in rank_grid.items():
                            if d_int == 0:  # Skip all D=0 points
                                continue
                            if k == 1 and s_int < 2:
                                continue
                            elif k == 2 and s_int < 4:
                                continue

                            fontsize = 13
                            fig_height_inches = fig.get_figheight()
                            axes_height_data = d_max - d_min
                            points_to_data = axes_height_data / (fig_height_inches * 72)
                            vertical_offset = -0.10 * fontsize * points_to_data

                            if rank == 'X':
                                ax.text(s_int, d_int + vertical_offset, r'\ding{55}',
                                       ha='center', va='center',
                                       fontsize=fontsize, fontweight='bold',
                                       zorder=11, clip_on=False)
                            elif rank is not None:
                                # Compute display label based on s-dependent rules
                                if s_int >= 3:
                                    display_label = 4 - rank
                                elif s_int == 2:
                                    display_label = 3 - rank
                                elif s_int == 1:
                                    display_label = 1 - rank
                                else:  # s_int == 0
                                    display_label = 0 - rank

                                ax.text(s_int, d_int + vertical_offset, str(display_label),
                                       ha='center', va='center',
                                       fontsize=fontsize, fontweight='bold',
                                       zorder=11, clip_on=False)

                    # Add colorbar
                    cbar_left = left3 + w3 + gap
                    cbar_ax = fig.add_axes([cbar_left, bottom_margin, cbar_w, height])
                    norm = plt.Normalize(vmin=0, vmax=1)
                    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='nipy_spectral'),
                                    cax=cbar_ax, orientation='vertical')
                    cb.set_ticks([0, 1])
                    cb.set_ticklabels(['0', '1'])
                    cb.ax.tick_params(labelsize=10, length=6, width=2)
                    # Match colorbar frame thickness to plot spines
                    try:
                        cb.outline.set_linewidth(2)
                    except Exception:
                        pass
                    for sp in cbar_ax.spines.values():
                        sp.set_linewidth(2)

                    out_png = os.path.join(frames_dir, f'frame_{f_idx:03d}.png')
                    plt.savefig(out_png, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    f_idx += 1
                    frame_paths.append(out_png)

        # Assemble GIF
        gif_path = '/home/barker/Documents/paper-v/Private/GeneralSolutionTransform.gif'
        try:
            from PIL import Image
            images = [Image.open(p) for p in frame_paths]
            # Duration calculated for 3 second cycle: 3000ms / num_frames
            duration_ms = int(1000 / len(images))
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
            print(f'GIF (smooth looped) saved as {gif_path} ({len(images)} frames @ {duration_ms}ms = {len(images)*duration_ms/1000:.1f}s per cycle)')
        except Exception as e:
            print(f'PIL unavailable or failed ({e}). GIF not created; frames in {frames_dir}')
    finally:
        plt.rcParams['text.usetex'] = prev_usetex
        try:
            if monitor_cpu and stop_mon:
                stop_mon()
        except Exception:
            pass

# Pre-compute discrete labels for both modes to detect differences
def compute_all_rank_grids():
    """Pre-compute rank grids for both Francia and Marzo modes"""
    rank_grids = {}
    
    cfg_discrete = PlotConfig()
    s_vals = range(int(cfg_discrete.s_min), int(cfg_discrete.s_max) + 1)
    d_vals = range(int(cfg_discrete.d_min), int(cfg_discrete.d_max) + 1)

    for mode in ["francia", "marzo"]:
        rank_grid = {}
        print(f"Pre-computing ranks for {mode} mode...")
        for s_int in s_vals:
            for d_int in d_vals:
                A_sym = build_matrix_A_symbolic_limit(s_int, d_int, mode=mode) if _HAVE_SYMPY else None
                A = A_sym if A_sym is not None else build_matrix_A(s_int, d_int, mode=mode)
                if A is None or np.any(np.isinf(A)) or np.any(np.isnan(A)):
                    rank_grid[(s_int, d_int)] = None
                else:
                    try:
                        rank = numerical_rank(A)
                        rank_grid[(s_int, d_int)] = rank
                    except Exception:
                        rank_grid[(s_int, d_int)] = None
        rank_grids[mode] = rank_grid
    
    return rank_grids

# Lazy computation of rank grids to avoid import-time cost
all_rank_grids = None
def get_all_rank_grids():
    global all_rank_grids
    if all_rank_grids is None:
        all_rank_grids = compute_all_rank_grids()
    return all_rank_grids

def _compute_francia_marzo_pdfs():
    grids = get_all_rank_grids()
    differences = set()
    for key in grids["francia"].keys():
        if grids["francia"][key] != grids["marzo"][key]:
            differences.add(key)
    print(f"Found {len(differences)} points with different ranks between Francia and Marzo modes")
    create_plots(mode="francia", rank_grid_override=grids["francia"], highlight_differences=differences)
    create_plots(mode="marzo", rank_grid_override=grids["marzo"], highlight_differences=differences)

if __name__ == "__main__":
    # ============================================================
    # PLOTTING MODE CONTROL - Select one of the following modes:
    # ============================================================

    PLOT_MODE = "francia"  # Options: "francia", "marzo", "both", "gif"

    # ============================================================
    # STYLE CUSTOMIZATION (optional)
    # ============================================================
    # To customize styling, uncomment and modify:
    # CUSTOM_STYLE = StyleConfig(
    #     # Font sizes
    #     title_fontsize=28,
    #     label_fontsize=22,
    #     tick_fontsize=18,
    #     rank_text_fontsize=14,
    #     # Line widths
    #     circle_linewidth=1.5,
    #     tick_width=2.5,
    #     spine_linewidth=2.5,
    #     # Colors
    #     circle_default_color='lightblue',
    #     circle_highlight_color='orange',
    #     # etc...
    # )

    # Mode-specific parameters
    GIF_PARAMS = {
        "num_frames": 300,
        "grid_n": 440,
        "seed": 9,
        "segments": 15,
        "monitor_cpu": False
    }

    # Execute selected mode
    if PLOT_MODE == "francia":
        print("Generating Francia mode PDF...")
        # To use custom styling:
        # cfg = PlotConfig(style=CUSTOM_STYLE)
        # create_plots(mode="francia")  # Then pass cfg if needed
        create_plots(mode="francia")

    elif PLOT_MODE == "marzo":
        print("Generating Marzo mode PDF...")
        create_plots(mode="marzo")

    elif PLOT_MODE == "both":
        print("Generating both Francia and Marzo mode PDFs with difference highlighting...")
        _compute_francia_marzo_pdfs()

    elif PLOT_MODE == "gif":
        print("Generating animated GIF...")
        create_transform_gif_smooth_looped(**GIF_PARAMS)

    else:
        raise ValueError(f"Unknown PLOT_MODE: {PLOT_MODE}. Use 'francia', 'marzo', 'both', or 'gif'.")
