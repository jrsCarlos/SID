import random

import numpy as np

from algs.solution_concepts import SolutionConcept
from algs.game_model import GameModel
import abc

class MARLAlgorithm(abc.ABC):
    @abc.abstractmethod
    def learn(self, joint_action, rewards, next_state: int, observations):
        pass

    @abc.abstractmethod
    def explain(self):
        pass

    @abc.abstractmethod
    def select_action(self, state):
        pass



class JALGT(MARLAlgorithm):
    def __init__(self, agent_id, game: GameModel, solution_concept: SolutionConcept,
                 gamma=0.95, alpha=0.5, epsilon=0.2, seed=42):
        self.agent_id = agent_id
        self.game = game                         # modelo del entorno
        self.solution_concept = solution_concept # como calcular politicas
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        # Q: N x S x AS
        # q_table: tabla de valores Q para cada (agente, estado, acción conjunta).
        # Dimensiones: n_agentes x n_estados x n_acciones_conjuntas. (N x S x AS)
        self.q_table = np.zeros((self.game.num_agents, self.game.num_states,
                                 len(self.game.action_space)))
        
        # Política conjunta por defecto: distribución uniforme respecto
        # de las acciones conjuntas, para cada acción (pi(a | s))

        # Inicializa una política uniforme: todos los agentes asignan igual probabilidad a todas las acciones en cada estado.
        # Política conjunta por defecto: distribución uniforme respecto
        # de las acciones conjuntas, para cada acción (pi(a | s))
        self.joint_policy = np.ones((self.game.num_agents, self.game.num_states,
                                     self.game.num_actions)) / self.game.num_actions
        self.metrics = {"td_error": []}

    def value(self, agent_id, state):
        value = 0
        # Recorremos todas las acciones conjuntas posibles
        for idx, joint_action in enumerate(self.game.action_space):

            # payoff: valor actual de la Q-table para el agente en el estado y acción conjunta
            payoff = self.q_table[agent_id][state][idx]

            # Probabilidad conjunta de que los agentes elijan esa accion
            joint_probability = np.prod([self.joint_policy[i][state][joint_action[i]]
                                         for i in range(self.game.num_agents)])
            value += payoff * joint_probability

        # Esperanza ponderada de los valores Q
        return value

    def update_policy(self, agent_id, state):
        self.joint_policy[agent_id][state] = self.solution_concept.solution_policy(agent_id, state, self.game,
                                                                                   self.q_table)

    def learn(self, joint_action, rewards, state, next_state):
        joint_action_index = self.game.action_space_index[joint_action]
        for agent_id in range(self.game.num_agents):
            # Obtenemos la recompensa del agente
            agent_reward = rewards[agent_id]

            # Calculamos el valor esperado del siguiente estado
            agent_game_value_next_state = self.value(agent_id, next_state)
            agent_q_value = self.q_table[agent_id][state][joint_action_index]

            # Calcula el error TD
            td_target = agent_reward + self.gamma * agent_game_value_next_state - agent_q_value

            # Actualizamos su Q-value
            self.q_table[agent_id][state][joint_action_index] += self.alpha * td_target
            self.update_policy(agent_id, state)

            # Guardamos el error de diferencia temporal para estadísticas posteriores
            self.metrics['td_error'].append(td_target)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def solve(self, agent_id, state):
        return self.joint_policy[agent_id][state]

    def select_action(self, state, train=True):
        # Si esta entrenando: usa politica ε-greedy.
        if train:
            if self.rng.random() < self.epsilon:
                return self.rng.choice(range(self.game.num_actions))
            else:
                probs = self.solve(self.agent_id, state)
                np.random.seed(self.rng.randint(0, 10000))
                return np.random.choice(range(self.game.num_actions), p=probs)
        # Si no: elige acción con máxima probabilidad
        else:
            return np.argmax(self.solve(self.agent_id, state))

    def explain(self, state=0):
        return self.solution_concept.debug(self.agent_id, state, self.game, self.q_table)
