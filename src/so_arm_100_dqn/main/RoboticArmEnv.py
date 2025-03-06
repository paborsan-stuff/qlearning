"""
Educational Use License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to use,
copy, modify, and distribute the Software solely for educational and
non-commercial purposes, subject to the following conditions:

[Include the full text as above or a reference to the LICENSE file]

This license is intended for educational purposes only.
"""

from robot_descriptions.loaders.mujoco import load_robot_description
import numpy as np
import mujoco
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from DQNAgent import DQNAgent
from pathlib import Path
import mediapy as media

#so_arm = load_robot_description("so_arm100_mj_description")
#so_arm.actuator

class RoboticArmEnv():
    
    def __init__(self, model, debug_mode):
        # Cargar modelo MuJoCo
        self.model = model
        self.data = mujoco.MjData(self.model)
        self.body_name = "Moving_Jaw"
        self.debug_mode = True
        # Espacio de observación (posiciones)
        self.state_dim = self.model.nq
        self.action_dim = self.model.nu

        # Obtener el ID del body "Moving_Jaw" (el que contiene el joint "Jaw")
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)

        # X posibles estados (posiciones) de un servo, definido por criterio propio
        self.num_actions = self.action_dim 

        # Agregar renderer para capturar imágenes
        self.renderer = mujoco.Renderer(self.model)
        with open('qlearning.json', 'r') as file:
            data = json.load(file)

        # Parámetros de DQN
        max_epsilon = data['max_epsilon']
        min_epsilon = data['min_epsilon']
        decay_rate = data['decay_rate']  # Decaimiento más lento para explorar más
        gamma = data['gamma']
        learning_rate = data['lr']
        memory_size = data['mem_size']  # Memoria más grande para mejorar el aprendizaje
        batch_size = data['batchsize']

        self.agent = DQNAgent(state_dim=self.state_dim,
                              num_actions=self.num_actions,
                              action_dim=self.action_dim,
                              gamma=gamma,
                              epsilon=max_epsilon,
                              epsilon_min=min_epsilon,
                              epsilon_decay=decay_rate,
                              learning_rate=learning_rate,
                              memory_size=memory_size,
                              batch_size=batch_size)

        # Posicion global (x,y,z) del 'Jaw' en Keyframe [0, -1.57079, 1.57079, 1.57079, -1.57079, 0]
        self.target_position = np.array([-0.0202, -0.23870058, 0.15226918]) 
        self.prev_jaw_position = np.array([0, 0, 0])  # Para calcular mejora en recompensa
        self.reset()


    def step(self, action, debug_mode = True):
        #
        # TODO Is it torque? or is pwm? It is angle
        #
        action_values = [(a - 1) * 0.1 for a in action]
        self.data.ctrl[:] = action_values
        mujoco.mj_step(self.model, self.data)

        # Obtener la posición absoluta del "Jaw" en coordenadas globales
        self.jaw_position = self.data.xpos[self.body_id]
        
        # Obtener observación
        obs = np.concatenate([self.data.qpos])

        # Calcular recompensa basada en distancia global (x,y,z) 
        distance = np.linalg.norm(self.target_position - self.jaw_position)
        prev_distance = np.linalg.norm(self.target_position - self.prev_jaw_position)
        improvement = prev_distance - distance

        reward = improvement * 10 - distance  # Premia acercarse, penaliza distancia

        if self.debug_mode:
            print("-" * 30)
            print(f"Target Position: {self.target_position}")
            print(f"Jaw Position: {self.jaw_position}")
            print(f"State: {self.data.ctrl[:]}")
            print(f"Action: {action_values}")
            print(f"Reward: {reward:.4f}")
            print("-" * 30)

        self.prev_qpos = self.data.qpos.copy()  # Guardar la posición actual
        done = distance <= 0.1  # Termina si alcanza la meta
        return obs, reward, done

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.prev_qpos = self.data.qpos.copy()  # Reiniciar posición previa
        return np.concatenate([self.data.qpos])

    def train(self, episodes=2):  # Entrenamos por más episodios
        success_history = []
        if self.debug_mode:
            print("-" * 30)
            print("Start training")
        for episode in tqdm(range(episodes)):
            state = self.reset()
            done = False
            total_reward = 0
            x = 0

            while not done:
                action = self.agent.get_action(state)
                next_state, reward, done = self.step(action)

                print(state)
                # Guardar experiencia en memoria
                self.agent.store_transition(state, action, reward, next_state, done)

                # Entrenar el agente
                self.agent.replay()

                state = next_state
                total_reward += reward

                # Renderizar y mostrar un frame cada 1000 episodios
                #if x % 10000 == 0:
                #    print(f"Rendering frame at episode {episode}...")
                #    self.renderer.update_scene(self.data)
                #    media.show_image(self.renderer.render())
                x += 1

            success_history.append(total_reward)

            if episode % 2 == 0:
                print(f"Episode {episode}: Reward {total_reward:.2f}")

        # Guardar modelo entrenado
        torch.save({
            'policy_net_state_dict': self.agent.policy_net.state_dict(),
            'target_net_state_dict': self.agent.target_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
        }, 'Checkpoint/dqn_robotic_arm.pth')

        # Graficar desempeño
        plt.plot(success_history)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Progress")
        plt.show()