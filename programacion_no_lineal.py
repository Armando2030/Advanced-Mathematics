import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x1, x2 = sp.symbols('x1 x2')
#f_symb = (1-x1)**2 + (x2-x1**2)**2
#f_symb = x1**2 + 25*x2**2
#f_symb = x1 - x2 + 2*x1*x2 + 2*x1**2 + x2**2
f_symb = 2*x1**2-1.05*x1**4+x1**6/6+x1*x2+x2**2
#f_symb = 0.26 * (x1**2 + x2**2) - 0.48*x1*x2
grad_f_symb = [sp.diff(f_symb, var) for var in (x1, x2)]
hess_f_symb = sp.hessian(f_symb, (x1, x2))
f_num = sp.lambdify((x1, x2), f_symb, 'numpy')
grad_f_num = sp.lambdify((x1, x2), grad_f_symb, 'numpy')
hess_f_num = sp.lambdify((x1, x2), hess_f_symb, 'numpy')

def f(x: np.ndarray) -> float:
    return f_num(x[0], x[1])

def gradiente(x: np.ndarray) -> np.ndarray:
    return np.array(grad_f_num(x[0], x[1]), dtype=float)

def hessiana(x: np.ndarray) -> np.ndarray:
    return np.array(hess_f_num(x[0], x[1]), dtype=float)

def busqueda_lineal(xk: np.ndarray, pk: np.ndarray) -> float:
    alpha = sp.symbols('alpha')
    xk_apk = xk + alpha * pk
    func_val = f(xk_apk)
    min_phi_alpha = sp.diff(func_val, alpha)
    alpha_values = sp.solve(min_phi_alpha, alpha)
    
    if alpha_values:
        func_min_value = float('inf')
        best_alpha = None
        for alpha_val in alpha_values:
            if alpha_val.is_real:
                func_val_at_alpha = func_val.subs(alpha, alpha_val)
                if func_val_at_alpha < func_min_value:
                    func_min_value = func_val_at_alpha
                    best_alpha = alpha_val
        if best_alpha is not None:
            return float(best_alpha)
    
    return 1.0

def gradiente_descendente(tol: float, nIter: int, x0: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    x_k = x0
    positions = [x0]
    for k in range(nIter):
        grad = gradiente(x_k)
        p_k = -grad
        if np.linalg.norm(grad) < tol: 
            print("Iteraciones necesarias:", k)
            return x_k, positions
        alpha_k = busqueda_lineal(x_k, p_k)  
        x_k = x_k + (alpha_k * p_k)
        positions.append(np.copy(x_k))
        print(f'Iteración {k+1}: x_k = {x_k}, grad = {grad}, p_k = {p_k}, norma = {np.linalg.norm(grad)}, alpha_k = {alpha_k}')
    return x_k, positions

def newton(tol: float, nIter: int, x0: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    x_k = x0
    positions = [x0]
    k = 0
    while k < nIter:
        hess_inv = np.linalg.inv(hessiana(x_k))
        p_k = -np.dot(hess_inv, gradiente(x_k))
        norma = np.linalg.norm(p_k)
        if norma <= tol:
            return x_k, positions
        alpha = busqueda_lineal(x_k, p_k)
        x_k = x_k + alpha * p_k
        positions.append(np.copy(x_k))
        k += 1
        grad = gradiente(x_k)  
        norma_grad = np.linalg.norm(grad)
        print(f'Iteración {k}: x_k = {x_k}, grad = {grad}, p_k = {p_k}, norma = {norma_grad}, alpha = {alpha}')
    return x_k, positions

def BFGS(tol: float, nIter: int, x0: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    x_k = x0
    positions = [x0]
    B_k = np.eye(len(x0)) 
    k = 0
    while k < nIter:
        grad = gradiente(x_k)
        p_k = -np.dot(B_k, grad)  
        if np.linalg.norm(grad) < tol:
            return x_k, positions
        alpha_k = busqueda_lineal(x_k, p_k)
        x_k1 = x_k + alpha_k * p_k  
        s_k = x_k1 - x_k
        y_k = gradiente(x_k1) - gradiente(x_k)
        if np.dot(y_k, s_k) > 0:  
            rho_k = 1.0 / np.dot(y_k.T, s_k)
            I = np.eye(len(B_k))
            term1 = (I - rho_k * np.outer(s_k, y_k.T)) @ B_k @ (I - rho_k * np.outer(y_k, s_k.T))
            term2 = rho_k * np.outer(s_k, s_k.T)
            B_k = term1 + term2   
        x_k = x_k1
        positions.append(np.copy(x_k))
        k += 1
        print(f'Iteración {k}: x_k = {x_k}, grad = {grad}, p_k = {p_k}, norma = {np.linalg.norm(grad)}, alpha = {alpha_k}')
    return x_k, positions

def powell(tolerancia: float, n_iter: int, x0: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    xk = x0
    positions = [x0]
    n = len(xk)
    directions = np.eye(n)
    k = 0
    while k < n_iter:
        Z = np.array(xk, dtype=float) 
        for i in range(n):
            si = directions[i]
            if f(xk + 0.01 * si) < f(xk):
                si = si
            else:
                si = -si
            alpha = busqueda_lineal(xk, si)
            xk += alpha * si
            positions.append(np.copy(xk))
        sn_plus_1 = xk - Z
        alpha = busqueda_lineal(xk, sn_plus_1)
        xk += alpha * sn_plus_1
        positions.append(np.copy(xk))
        if np.linalg.norm(xk - (Z + alpha * sn_plus_1)) < tolerancia:
            return xk, positions
        directions[:-1] = directions[1:]
        directions[-1] = sn_plus_1 / np.linalg.norm(sn_plus_1)
        k += 1
        print(f"Iteración {k}: Punto actual: {xk}")
    return xk, positions

def grafica() -> None:
    x1_1 = np.linspace(-5, 5, 100)
    x2_1 = np.linspace(-5, 5, 100)
    X1_1, X2_1 = np.meshgrid(x1_1, x2_1)
    Z1 = f((X1_1, X2_1))
    x1_2 = np.linspace(-2, 2, 100)
    x2_2 = np.linspace(-1, 1, 100)
    X1_2, X2_2 = np.meshgrid(x1_2, x2_2)
    Z2 = f((X1_2, X2_2))
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X1_1, X2_1, Z1, cmap='copper')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Three-Hump Camel Function', fontsize=10)
    ax1.view_init(elev=30, azim=50)  
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X1_2, X2_2, Z2, cmap='copper')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('Three-Hump Camel Function', fontsize=10)
    ax2.view_init(elev=30, azim=45)
    plt.show()

def direcciones_punto(f: callable, positions: list[np.ndarray], title: str) -> None:
    x1_vals = np.linspace(-5, 5, 800) 
    x2_vals = np.linspace(-5, 5, 800)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f((X1, X2))
    plt.figure(figsize=(10, 8)) 
    plt.contour(X1, X2, Z, levels=np.logspace(-0.5, 3, 35), cmap='viridis')
    plt.plot(*zip(*positions), marker='o', color='r', markersize=5)
    plt.title(title, fontsize=14) 
    plt.xlabel('x1', fontsize=12)
    plt.ylabel('x2', fontsize=12)
    plt.xlim([-5, 5]) 
    plt.ylim([-5, 5])
    plt.show()

def menu():
    print("Selecciona un método")
    print("1. Gradiente Descendente")
    print("2. Newton")
    print("3. Cuasi Newton (BFGS)")
    print("4. Powell")
    num_metodo = int(input("Ingresa el número que corresponde al método: "))
    tol = float(input("Ingresa la tolerancia en decimal: "))
    nIter = int(input("Ingresa el número máximo de iteraciones: "))
    x0 = np.array([float(x) for x in input("Ingresa el valor de cada entrada del vector x0 separados por un espacio: ").split()])
    metodos = {
        1: gradiente_descendente,
        2: newton,
        3: BFGS,
        4: powell
    }
    metodo = metodos.get(num_metodo)
    if metodo:
        punto_optimo, posiciones = metodo(tol, nIter, x0)
        print("Punto óptimo: ", punto_optimo)
        #direcciones_punto(f, posiciones, f'Método {num_metodo}')