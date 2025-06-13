import os
import time
import json
import argparse
import numpy as np
from tqdm import tqdm
from gymnasium import Wrapper
from algs.utils import draw_history
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor, AnimationConfig
from algs.algorithms import JALGT
from algs.game_model import GameModel
from algs.solution_concepts import MinimaxSolutionConcept, ParetoSolutionConcept, NashSolutionConcept, WelfareSolutionConcept


#=============================================================================================#
#                               OBSERVACION PARCIAL COMO ESTADO                               #
#=============================================================================================#

def obs_to_state(obs):
    matrix_obstacles = obs[0]
    matrix_agents = obs[1]
    matrix_target = obs[2]

    # Representación del objetivo:
    #  Ocupa 2 bits
    #  0 si el objetivo está arriba, diagonal arriba-izquierda o diagonal arriba-derecha
    #  1 si el objetivo está abajo, diagonal abajo-izquierda o diagonal abajo-derecha
    #  2 si el objetivo está a la izquierda (no en diagonal)
    #  3 si el objetivo está a la derecha (no en diagonal)
    target = np.max(matrix_target[2]) * 1 + \
             matrix_target[1][0] * 2 + matrix_target[1][2] * 3

    # Representación de los obstáculos:
    #  Shift de 2^6, ocupando 4 bits
    #  2^9 si hay un obstáculo arriba (no diagonal)
    #  2^8 si hay un obstáculo a la izquierda (no diagonal)
    #  2^7 si hay un obstáculo a la derecha (no diagonal)
    #  2^6 si hay un obstáculo abajo (no diagonal)
    obstacles = matrix_obstacles[0][1] * 2 ** 9 + \
                matrix_obstacles[1][0] * 2 ** 8 + \
                matrix_obstacles[1][2] * 2 ** 7 + \
                matrix_obstacles[2][1] * 2 ** 6

    # Representación de los otros agentes:
    #  Shift de 2^2, ocupando 4 bits
    #  2^5 si hay un agente arriba (no diagonal)
    #  2^4 si hay un agente a la izquierda (no diagonal)
    #  2^3 si hay un agente a la derecha (no diagonal)
    #  2^2 si hay un agente abajo (no diagonal)
    agents = matrix_agents[0][1] * 2 ** 5 + \
             matrix_agents[1][0] * 2 ** 4 + \
             matrix_agents[1][2] * 2 ** 3 + \
             matrix_agents[2][1] * 2 ** 2

    return int(obstacles + agents + target)

#=============================================================================================#
#                                    FUNCION DE RECOMPENSA                                    #
#=============================================================================================#
class RewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, joint_action):
        # En caso de que queráis utilizar las observaciones anteriores, utilizad este objeto:
        previous_observations = self.env.unwrapped._obs()

        observations, rewards, terminated, truncated, infos = self.env.step(joint_action)
        for i in range(len(joint_action)):
            if not terminated[i] and not truncated[i]:
                if rewards[i] == 0:  # Penalización por tardar más en llegar
                    rewards[i] = rewards[i] - 0.01
        return observations, rewards, terminated, truncated, infos


def create_env(config, seed=42):
    grid_config = GridConfig(num_agents=config["num_agents"],
                             size=config["size"],
                             density=config["obstacle_density"],
                             seed=seed,
                             max_episode_steps=config["episode_length"],
                             obs_radius=config["obs_radius"],
                             on_target="finish",
                             render_mode=None)
    animation_config = AnimationConfig(directory='renders/',  # Dónde se guardarán las imágenes
                                       static=False,
                                       show_agents=True,
                                       egocentric_idx=None,  # Punto de vista
                                       save_every_idx_episode=config["save_every"],  # Guardar cada save_every episodios
                                       show_border=True,
                                       show_lines=True)
    env = pogema_v0(grid_config)
    env = AnimationMonitor(env, animation_config=animation_config)
    return RewardWrapper(env)  # Añadimos nuestra función de recompensa

#=============================================================================================#
#                                CONFIGURACION DEL EXPERIMENTO                                #
#=============================================================================================#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  type=str, required=True)
    parser.add_argument('--datadir', type=str, required=True)
    args = parser.parse_args()

    # Mapeo de nombres a clases de conceptos de solución
    SOLUTION_CONCEPTS = {
        "Pareto": ParetoSolutionConcept,
        "Nash": NashSolutionConcept,
        "Minimax": MinimaxSolutionConcept,
        "Welfare": WelfareSolutionConcept
    }

    # Cargar configuración desde un archivo JSON
    with open(args.config, "r") as f:
        exp_config = json.load(f)

    # Convertir el nombre del concepto de solución a una clase
    solution_concept_str = exp_config.get("solution_concept", "Pareto")
    exp_config["solution_concept"] = SOLUTION_CONCEPTS.get(solution_concept_str, NashSolutionConcept)
    exp_config["save_every"] = exp_config.get("save_every", 10)
    try:
        os.mkdir(exp_config["renders"])
    except:
        pass


    #=============================================================================================#
    #                                  ENTRENAMIENTO Y EVALUACION                                 #
    #=============================================================================================#

    ##### DECLARACION DEL MODELO DE JUEGO Y DE LOS ALGORITMOS PARA CADA AGENTE #####
    game = GameModel(num_agents=exp_config["num_agents"], num_states=exp_config["num_states"],
                     num_actions=5)  # STAY, UP, DOWN, LEFT, RIGHT
    
    algorithms = [JALGT(i, game,
                        exp_config["solution_concept"](),
                        gamma=exp_config["gamma"],
                        alpha=exp_config["learning_rate"],
                        epsilon=exp_config["epsilon_max"],
                        seed=i) for i in range(game.num_agents)]


    # Variable auxiliar para indicar el decremento de epsilon después de cada episodio.
    # Caída lineal de epsilon: precalculamos la diferencia en cada paso
    epsilon_diff = (exp_config["epsilon_max"] - exp_config["epsilon_min"]) / exp_config["episodes_per_epoch"]

    # Variables auxiliares para almacenar métricas
    INDIVIDUAL_REWARD_PER_EPOCH = []
    COLECTIVE_REWARD_PER_EPOCH = []
    TD_ERROR_PER_EPOCH = []
    TIME_PER_EPISODE = []

    TOTAL_TRAINING_TIME = 0
    TOTAL_EPISODES = 0

    ##### ENTRENAMIENTO Y EVALUACIÓN #####
    pbar = tqdm(range(exp_config["epochs"]))  # Barra de progreso
    total_training_start_time = time.time()
    
    for epoch in pbar:
        all_eval_rewards = []
        all_td_errors = []

        epoch_start_time = time.time()

        #=============== ENTRENAMIENTO ===============#
        for ep in range(exp_config["episodes_per_epoch"]):
            pbar.set_postfix({'modo': 'entrenamiento', 'episodio': ep})
            env = create_env(config=exp_config, seed=ep % exp_config["maps"])
            observations, infos = env.reset()
            terminated = truncated = [False, ...]
            train_rewards = [0] * game.num_agents
            states = [obs_to_state(observations[i]) for i in range(game.num_agents)]

            while not all(terminated) and not all(truncated):  # Hasta que acabe el episodio
                # Elegimos acciones
                actions = tuple([algorithms[i].select_action(states[i]) for i in range(game.num_agents)])
                # Ejecutamos acciones en el entorno
                observations, rewards, terminated, truncated, infos = env.step(actions)
                # Aprendemos: actualizamos valores Q
                [algorithms[i].learn(actions, rewards, states[i], obs_to_state(observations[i]))
                 for i in range(game.num_agents)]
                # Actualizamos métricas
                train_rewards = [train_rewards[i] + rewards[i] for i in range(game.num_agents)]
                all_td_errors.append(algorithms[0].metrics["td_error"][-1])
                # Preparar siguiente iteración: convertir observaciones parciales en estados
                states = [obs_to_state(observations[i]) for i in range(game.num_agents)]

            TOTAL_EPISODES += 1
            # Actualizamos epsilon
            [algorithms[i].set_epsilon(exp_config["epsilon_max"] - epsilon_diff * ep) for i in range(game.num_agents)]

        elapsed = time.time() - epoch_start_time
        TOTAL_TRAINING_TIME += elapsed
        TIME_PER_EPISODE.append(elapsed)
        TD_ERROR_PER_EPOCH.append(sum(all_td_errors))

        #=============== EVALUACION ===============#
        evaluation_episodes = exp_config["maps"]
        all_eval_rewards = []
        agent0_reward_epoch = 0

        for ep in range(evaluation_episodes):
            pbar.set_postfix({'modo': 'evaluación...', 'episodio': ep})
            env = create_env(config=exp_config, seed=ep)  # Reaprovechamos mapas del entrenamiento
            observations, infos = env.reset()
            terminated = truncated = [False, ...]
            total_rewards = [0] * exp_config["num_agents"]
            states = [obs_to_state(observations[i]) for i in range(game.num_agents)]

            while not all(terminated) and not all(truncated):  # Hasta que acabe el episodio
                states = [obs_to_state(observations[i]) for i in range(game.num_agents)]
                actions = tuple([algorithms[i].select_action(states[i], train=False)
                                 for i in range(game.num_agents)])
                observations, rewards, terminated, truncated, infos = env.step(actions)
                total_rewards = [total_rewards[i] + rewards[i] for i in range(exp_config["num_agents"])]
            
            # Guardamos animaciones
            for agent_i in range(exp_config["num_agents"]):
                solution_concept_name = exp_config["solution_concept"].__name__
                env.save_animation(f"{exp_config['renders']}/{solution_concept_name}-map{ep}-agent{agent_i}-epoch{epoch}.svg",
                                   AnimationConfig(egocentric_idx=agent_i, show_border=True, show_lines=True))
            all_eval_rewards.append(sum(total_rewards))
            agent0_reward_epoch += total_rewards[0]

        INDIVIDUAL_REWARD_PER_EPOCH.append(agent0_reward_epoch)
        COLECTIVE_REWARD_PER_EPOCH.append(sum(all_eval_rewards))
        pbar.set_description(f"Recompensa colectiva del último epoch = {'{:>6.6}'.format(str(sum(all_eval_rewards)))}")


    ##### GRAFICAS CON EL RESULTADO DE LA RECOLECCION DE METRICAS #####
    draw_history(INDIVIDUAL_REWARD_PER_EPOCH, "Recompensa del Agente 0", save_dir="plots/" + args.datadir)
    draw_history(COLECTIVE_REWARD_PER_EPOCH, "Recompensa colectiva", save_dir="plots/" + args.datadir)
    draw_history(TD_ERROR_PER_EPOCH, "TD Error", save_dir="plots/" + args.datadir)

    resultados = {
        "config": exp_config.copy(),
        "metrics": {
            "individual_reward_per_epoch": INDIVIDUAL_REWARD_PER_EPOCH,
            "collective_reward_per_epoch": COLECTIVE_REWARD_PER_EPOCH,
            "td_error_per_epoch": TD_ERROR_PER_EPOCH,
            "time_per_episode": TIME_PER_EPISODE,
            "total_training_time": TOTAL_TRAINING_TIME,
            "total_episodes": TOTAL_EPISODES
        }
    }

    # Convertimos la clase a string para que sea serializable
    if hasattr(resultados["config"]["solution_concept"], "__name__"):
        resultados["config"]["solution_concept"] = resultados["config"]["solution_concept"].__name__

    fileName = str(exp_config["num_agents"]) + "-agents" + "_" + str(exp_config["size"]) + "-size" + "_" + str(exp_config["obstacle_density"]) + "-density" + "_" + str(exp_config["epochs"]) + "-epochs" + "_" + str(exp_config["episodes_per_epoch"]) + "-episodes" + "_" + str(exp_config["gamma"]) + "-gamma" + "_" + str(exp_config["learning_rate"]) + "-alpha" + "_" + str(exp_config["epsilon_max"]) + "-epsilon.json"
    ouputfile = "data/" + args.datadir + "/" + fileName
    with open(ouputfile, "w") as f:
        json.dump(resultados, f, indent=4)

