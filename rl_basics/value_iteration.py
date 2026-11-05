"""Value iteration in a deterministic 3x3 gridworld."""

STATES = [(row, col) for row in range(3) for col in range(3)]
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
GOAL = (0, 2)


def transition(state, action):
    if state == GOAL:
        return state, 0
    dr, dc = ACTIONS[action]
    next_state = (min(2, max(0, state[0] + dr)), min(2, max(0, state[1] + dc)))
    return next_state, 10 if next_state == GOAL else 0


def action_value(state, action, values, gamma):
    next_state, reward = transition(state, action)
    return reward + gamma * values[next_state]


def value_iteration(gamma=0.9, tolerance=1e-8):
    values = {state: 0.0 for state in STATES}
    while True:
        change = 0.0
        for state in STATES:
            if state == GOAL:
                continue
            old = values[state]
            values[state] = max(action_value(state, action, values, gamma) for action in ACTIONS)
            change = max(change, abs(values[state] - old))
        if change < tolerance:
            break
    policy = {}
    for state in STATES:
        if state == GOAL:
            continue
        policy[state] = max(ACTIONS, key=lambda action: action_value(state, action, values, gamma))
    return values, policy


if __name__ == "__main__":
    values, policy = value_iteration()
    for row in range(3):
        print("  ".join(f"{values[row, col]:5.2f}" for col in range(3)))
    print(policy)
