import math

def find_general_hyperband_schedule(B, eta_values, R_min, R_max, n_min):
    best_schedules = []
    for eta in eta_values:
        # Determine number of rounds
        rounds = math.floor(math.log(R_max / R_min, eta)) +1
        # Compute resource per prompt per round
        R_list = [int(R_min * eta ** i) for i in range(rounds)]
        # Compute multiplicity per round
        m_list = [rounds - i for i in range(rounds)]
        # Compute eta multipliers for prompt counts
        eta_powers = [eta ** (rounds -1 - i) for i in range(rounds)]
        # Compute cost coefficients per x_last
        coeffs = [4 * m * R * e for m, R, e in zip(m_list, R_list, eta_powers)]
        total_coeff = sum(coeffs)
        # Maximize x_last under constraints
        max_x_last = B // total_coeff
        if max_x_last < n_min:
            continue  # skip infeasible
        x_last = max_x_last
        x_list = [x_last * eta ** (rounds -1 - i) for i in range(rounds)]
        # Total cost
        total_cost = sum(c * x_last for c in coeffs)
        schedule = {
            'eta': eta,
            'rounds': rounds,
            'x_list': x_list,
            'R_list': R_list,
            'multiplicity': m_list,
            'total_cost': total_cost
        }
        best_schedules.append(schedule)
    return best_schedules

def time_to_budget(hours):
    return (hours * 3600) / 4