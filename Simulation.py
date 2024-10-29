import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.optimize import newton

# Definindo constantes físicas
m, g = 1, 9.81
a, b = 100, 100

# Função para encontrar theta2 usando o método de Newton
def f(theta):
    return b / a - (1 - np.cos(theta)) / (theta - np.sin(theta))

# Calcular theta2 e o raio R correspondente
theta2 = newton(f, np.pi / 2)
R = b / (1 - np.cos(theta2)) 

# Função que representa a EDO para a cicloide
def derivada(t, x):
    k = (np.sqrt(2 * g * R)) / R
    return np.sqrt(g*m / R)

# Função que representa a EDO para o plano inclinado linear
def derivada_linear(t, x):
    y_x = x * b / a
    dy_x = b / a
    return np.sqrt(2 * g * y_x) / np.sqrt(1 + dy_x**2)

# Condição inicial para a solução da EDO
x0 = 0.001  # x(0) = 0.1 para evitar divisão por zero

# Intervalo de tempo para a solução e pontos de avaliação
t_span = (0, 10)  # Intervalo de tempo de 0 a 10 segundos
t_eval = np.linspace(*t_span, num=1000)  # 1000 pontos no intervalo

# Resolver a EDO para a cicloide usando o método 'BDF'
solucao = solve_ivp(derivada, t_span, [x0], t_eval=t_eval)

# Resolver a EDO para o plano inclinado linear usando o método 'BDF'
solucao_linear = solve_ivp(derivada_linear, t_span, [x0], t_eval=t_eval, method='BDF')
x_linear = solucao_linear.y[0]
y_linear = b / a * x_linear  # Calcular as coordenadas do plano inclinado linear

# Extrair os tempos e valores de theta da solução
t = solucao.t
theta = solucao.y[0]

# Funções para calcular as coordenadas x e y da cicloide
def x_cycloide(theta):
    return R * (theta - np.sin(theta))

def y_cycloide(theta):
    return R * (1 - np.cos(theta))

# Calcular as coordenadas da cicloide usando os valores de theta
x_c = x_cycloide(theta)
y_c = y_cycloide(theta)

# Configuração da animação
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, a + 1)
ax.set_ylim(-0.1, b + 20)
ax.scatter(a,b , marker = 'o' , color = 'green', label = 'ponto (a,b) final')
ax.invert_yaxis()
ax.set_title('Simulação de Trajetórias')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.grid(True)  # Adiciona a grade ao gráfico

line_linear, = ax.plot([], [], 'b-', label='Reta')
line_cycloid, = ax.plot([], [], 'r-', label='Cicloide')
point_linear, = ax.plot([], [], 'bo')
point_cycloid, = ax.plot([], [], 'ro')

# Função de inicialização da animação
def init():
    line_linear.set_data([], [])
    line_cycloid.set_data([], [])
    point_linear.set_data([], [])
    point_cycloid.set_data([], [])
    return line_linear, line_cycloid, point_linear, point_cycloid

# Função de atualização da animação
def update(frame):
    line_linear.set_data(x_linear[:frame], y_linear[:frame])
    line_cycloid.set_data(x_c[:frame], y_c[:frame])
    point_linear.set_data(x_linear[frame], y_linear[frame])
    point_cycloid.set_data(x_c[frame], y_c[frame])
    return line_linear, line_cycloid, point_linear, point_cycloid

# Criação da animação
anim = FuncAnimation(
    fig, update, frames=range(0, len(x_c), 2),
    init_func=init, blit=True, interval=10
)


ax.legend(loc='upper right')
plt.show()
