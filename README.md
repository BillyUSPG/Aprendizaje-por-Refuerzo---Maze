🔹 Aprendizaje por Refuerzo

Es una forma de aprendizaje automático en la que un agente aprende a tomar decisiones dentro de un entorno. En lugar de que alguien le diga la respuesta correcta, el agente prueba acciones, recibe recompensas o castigos, y poco a poco mejora su comportamiento.

🔹 Representación del Laberinto

El entorno se representa como un laberinto de casillas:

0 son caminos libres.

1 son paredes.

La salida está en la esquina final.

El agente aparece en una casilla de inicio y se mueve paso a paso hasta intentar llegar a la meta.

🔹 Acciones posibles

En cada paso, el agente puede elegir moverse en cuatro direcciones:

Arriba (↑)

Abajo (↓)

Izquierda (←)

Derecha (→)

🔹 Recompensas y penalizaciones

El sistema de aprendizaje se guía con recompensas:

+1 cuando avanza a una casilla libre.

-5 si choca contra una pared.

+50 (muy positivo) cuando llega a la meta.

Esto motiva al agente a buscar la salida y evitar errores.

🔹 El agente que aprende (Q-learning)

El agente empieza sin saber nada, se mueve al azar y comete muchos errores.
Con el tiempo, va guardando en una tabla (Q-table) el valor de cada acción en cada casilla.

Si una acción lo acerca a la meta, esa acción se vuelve más atractiva.

Si una acción lo hace chocar, se penaliza y el agente la evita en el futuro.

Después de varios intentos, el agente aprende la mejor ruta para llegar a la salida de manera eficiente.
