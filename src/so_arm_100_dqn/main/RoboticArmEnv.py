"""
Educational Use License

Copyright (c) 2025 UP Student IA Research

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
import enum
from RobotAction import RobotAction 

class RoboticArmEnv():
    
    def __init__(self, model, debug_mode, steps, max_ep):
        # Cargar modelo MuJoCo

        self.model = model
        self.data = mujoco.MjData(self.model)
        self.body_name = "Moving_Jaw"
        self.debug_mode = debug_mode
        # Espacio de observación (posiciones)
        self.state_dim = self.model.nq
        self.action_dim = self.model.nu
        self.steps = steps
        self.max_ep = max_ep
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
        robot_trainer = RobotAction(model)

        self.agent = DQNAgent(state_dim=self.state_dim,
                              num_actions=self.num_actions,
                              action_dim=self.action_dim,
                              gamma=gamma,
                              epsilon=max_epsilon,
                              epsilon_min=min_epsilon,
                              epsilon_decay=decay_rate,
                              learning_rate=learning_rate,
                              memory_size=memory_size,
                              batch_size=batch_size,
                              dq_trainer = robot_trainer)

        # Posicion global (x,y,z) del 'Jaw' en Keyframe [0, -1.57079, 1.57079, 1.57079, -1.57079, 0]
        self.target_position = np.array([-0.0202, -0.23870058, 0.15226918]) 
        self.prev_jaw_position = np.array([0, 0, 0])  # Para calcular mejora en recompensa
        self.reset()
    def step(self, action, debug_mode = True):


        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # Obtener la posición absoluta del "Jaw" en coordenadas globales
        self.jaw_position = self.data.xpos[self.body_id]
        
        # Obtener observación
        obs = np.concatenate([self.data.qpos])

        # Calcular recompensa basada en distancia global (x,y,z) 
        current_distance = np.linalg.norm(self.target_position - self.jaw_position)
        prev_distance = np.linalg.norm(self.target_position - self.prev_jaw_position)

        #          condition                        reward       reward
        # prev_distance > current_distance            +            1
        # prev_distance < current_distance            -           -1
        #
        # max(reward)
        #
        # Max = (10pd - 11cd)
        #
        reward = (prev_distance - current_distance) * 10

        if self.debug_mode:
            print("-" * 30)
            print(f"Target Position: {self.target_position}")
            print(f"Jaw Position: {self.jaw_position}")
            print(f"State: {self.data.ctrl[:]}")
            print(f"Action: {action}")
            print(f"Reward: {reward:.4f}")
            print("-" * 30)

        self.prev_qpos = self.data.qpos.copy()  # Guardar la posición actual
        done = current_distance <= 0.1  # Termina si alcanza la meta
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
        for episode in tqdm(range(self.max_ep)):
            state = self.reset()
            done = False
            total_reward = 0
            step = 0
               
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
                step += 1
                if step > self.steps:
                    break
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

    def unit_smooth(self, normalised_time: float) -> float:
        return 1 - np.cos(normalised_time * 2 * np.pi)

    def azimuth(self, 
        time: float, duration: float, total_rotation: float, offset: float
    ) -> float:
        return offset + self.unit_smooth(time / duration) * total_rotation
        
    def quartic(self, t: float) -> float:
        return 0 if abs(t) > 1 else (1 - t**2) ** 2
    
    def blend_coef(self, t: float, duration: float, std: float) -> float:
        normalised_time = 2 * t / duration - 1
        return self.quartic(normalised_time / std)

    def inference(self):
        # Cargar el modelo guardado
        checkpoint = torch.load("Checkpoint/dqn_robotic_arm.pth", map_location=torch.device("cpu"))
        self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.agent.epsilon = 0
        self.fps = 60
        self.duration = 10.0
        self.total_rot = 60
        self.blend_std = .8

        # Creando visuales y colisiones.
        vis = mujoco.MjvOption()
        vis.geomgroup[2] = True
        vis.geomgroup[3] = False
        coll = mujoco.MjvOption()
        coll.geomgroup[2] = False
        coll.geomgroup[3] = True
        coll.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True

        # Enfoque de cámara.
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, camera)
        camera.distance = 1
        offset = self.model.vis.global_.azimuth

        class Resolution(enum.Enum): #perdón Tonix
            SD = (480, 640)
            HD = (720, 1280)
            UHD = (2160, 3840)

        res = Resolution.SD
        h, w = res.value

        renderer = mujoco.Renderer(self.model, height=h, width=w)

        mujoco.mj_resetData(self.model, self.data)  # Resetear el entorno

        upwards = 0
        frames = []
        for step in range(1000):  # Simular 1000 pasos
            state = np.concatenate([self.data.qpos])  # Obtener el estado
            action = self.agent.get_action(state)  # Obtener acción del modelo cargado

            # Convertir la acción discreta en torques
            self.data.ctrl[:] = action  # Aplicar acción al simulador

            mujoco.mj_step(self.model, self.data)  # Avanzar la simulación
            if len(len.frames) < self.data.time * self.fps:
                camera.azimuth = self.azimuth(self.data.time, self.duration, self.total_rot, offset)
                renderer.update_scene(self.data, camera, scene_option=vis)
                vispix = renderer.render().copy().astype(np.float32)
                renderer.update_scene(self.data, camera, scene_option=coll)
                collpix = renderer.render().copy().astype(np.float32)
                b = self.blend_coef(self.data.time, self.duration, self.blend_std)
                frame = (1 - b) * vispix + b * collpix
                frame = frame.astype(np.uint8)
                frames.append(frame)
                upwards =+ 1
        media.show_video(frames, fps=self.fps, loop=False)