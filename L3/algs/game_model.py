import itertools


class GameModel:
    def __init__(self, num_agents, num_states, num_actions):
        self.num_agents = num_agents
        self.num_states = num_states # número total de estados posibles del entorno, usado para la Q-table
        # Conjunto de acciones conjuntas posibles. 
        # Por ejemplo, si hay 2 agentes y cada uno tiene 2 acciones
        # las acciones conjuntas serían: (0,0), (0,1), (1,0), (1,1).
        self.num_actions = num_actions
        self.action_space = self.generate_action_space()
        self.action_space_index = {joint_action: idx for idx, joint_action in enumerate(self.action_space)}

    def generate_action_space(self):
        actions_by_players = []
        for agent_id in range(self.num_agents):
            actions_by_players.append(range(self.num_actions))
        all_joint_actions = itertools.product(*actions_by_players)
        return [tuple(l) for l in all_joint_actions]
