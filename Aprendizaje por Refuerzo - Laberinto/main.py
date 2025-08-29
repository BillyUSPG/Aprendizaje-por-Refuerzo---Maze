import logging  
import numpy as np  

import matplotlib.pyplot as plt 

import models  
from environment.maze import Maze, Render  

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)  

class Options:
    SHOW_MAZE_ONLY = 1  
    Q_ELIGIBILITY = 2   
    SALIR = 3          

mazes = {
    1: np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0]
    ]),
    2: np.array([
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 2]
    ]),
    3: np.array([
        [0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 2]
    ]),
    4: np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 2]
    ])
}


while True:
    print("\nOpciones:")  
    print("1 - Mostrar solo el laberinto")
    print("2 - Ejecutar modelo Aprendizaje por Refuerzo")
    print("3 - Salir")

    opcion = int(input("Selecciona una opción (1-3): "))  

    if opcion == Options.SALIR:
        print("¡Hasta luego!")  
        break  

    maze_id = int(input("Selecciona un laberinto (1-4): "))  
    maze = mazes.get(maze_id, mazes[1])  
    game = Maze(maze)  

    if opcion == Options.SHOW_MAZE_ONLY:
        game.render(Render.MOVES)  
        game.reset()               
        plt.show()                 

    elif opcion == Options.Q_ELIGIBILITY:
        game.render(Render.TRAINING)  
        model = models.QTableTraceModel(game)  

        h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
                                 stop_at_convergence=True)
 
        fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
        fig.canvas.manager.set_window_title(model.name)  
        ax1.plot(*zip(*w))  
        ax1.set_xlabel("episode")
        ax1.set_ylabel("win rate")
        ax2.plot(h) 
        ax2.set_xlabel("episode")
        ax2.set_ylabel("cumulative reward")
        plt.show() 
