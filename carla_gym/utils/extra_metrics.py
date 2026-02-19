from scipy.optimize import fsolve


def compute_Icell(v, a):
    """
    Compute cell current Icell (A).
    Input: v (m/s) vehicle speed, a (m/s²) acceleration.
    Output: Icell_t (A).
    """
    m, g, Cr, Cd, rho_a, A = 1847, 9.81, 0.01, 0.23, 1.226, 2.2
    delta_r, R_wh, i_g, eta_EM, eta_bat = 1.1, 0.33, 9.0, 0.9, 0.95
    n_ser, n_par, Voc, Rint = 100, 10, 370, 0.1
    Ft = m * g * Cr + 0.5 * rho_a * Cd * A * v**2 + delta_r * m * a

    TEM = Ft * R_wh / i_g
    omega_EM = v / R_wh * i_g

    kappa = 1 if TEM >= 0 else -1
    PEM = omega_EM * TEM * (eta_EM ** -kappa)

    Pcell = PEM / eta_bat / (n_ser * n_par)

    # Solve nonlinear eq for Icell
    def equation(I):
        return I * (Voc - I * Rint) - Pcell

    I_guess = Pcell / Voc
    Icell_t = fsolve(equation, I_guess)[0]

    return Icell_t


def compute_fuel_rate(v, a):
    """Compute fuel consumption rate (kg/s). Input: v (m/s), a (m/s²)."""
    m, g, Cr, Cd, rho, A = 1847, 9.81, 0.01, 0.23, 1.226, 2.2
    delta_r, eta_engine, LHV = 1.05, 0.3, 44e6
    Ft = m * g * Cr + 0.5 * rho * Cd * A * v ** 2 + delta_r * m * a

    P_wheel = Ft * v
    if P_wheel <= 0:
        fuel_rate = 0.0
    else:
        P_engine = P_wheel / eta_engine
        fuel_rate = P_engine / LHV

    return fuel_rate


