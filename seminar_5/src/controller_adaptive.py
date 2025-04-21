import numpy as np

class AdaptiveController:
    """
    Адаптивный контроллер для системы маятника.

    Оценивает неизвестный коэффициент трения C и использует его
    для вычисления управляющего момента.
    """
    def __init__(self, m, l, g, max_torque, alpha, initial_state, dt):
        """
        Инициализация контроллера.

        Args:
            m (float): Масса маятника.
            l (float): Длина маятника.
            g (float): Ускорение свободного падения.
            max_torque (float): Максимальный допустимый момент (tau_bar).
            alpha (float): Скорость обучения для оценки трения.
            initial_state (np.ndarray): Начальное состояние [theta, theta_dot].
            dt (float): Шаг времени симуляции.
        """
        self.m = m
        self.l = l
        self.g = g
        self.max_torque = max_torque  # tau_bar
        self.alpha = alpha
        self.dt = dt

        # Начальная оценка коэффициента трения
        self.C_hat = 0.0

        # Желаемая энергия (верхнее положение)
        self.E_des = 2 * self.m * self.g * self.l

        print(f"Adaptive controller created with m={m}, l={l}, g={g}, max_torque={max_torque}, alpha={alpha}")
        print(f"Initial C_hat = {self.C_hat}")
        print(f"Desired Energy E_des = {self.E_des}")

    def _calculate_energy(self, state):
        """Вычисляет текущую полную энергию системы."""
        theta, theta_dot = state
        E_kin = 0.5 * self.m * self.l**2 * theta_dot**2
        E_pot = self.m * self.g * self.l * (1 - np.cos(theta))
        return E_kin + E_pot

    def compute_control(self, state):
        """
        Вычисляет управляющий момент и обновляет оценку трения.

        Args:
            state (np.ndarray): Текущее состояние [theta, theta_dot].

        Returns:
            float: Вычисленный управляющий момент (tau).
        """
        theta, theta_dot = state
        S1 = theta
        S2 = theta_dot

        # Вычисляем ошибку энергии
        E_tot = self._calculate_energy(state)
        delta_E = self.E_des - E_tot

        # Вычисляем управляющий момент 'a' (tau)
        # a = tau_bar * sgn(delta_E * S2) + C_hat * m * l^2 * |S2| * S2
        term1 = self.max_torque * np.sign(delta_E * S2)
        term2 = self.C_hat * self.m * self.l**2 * np.abs(S2) * S2
        torque = term1 + term2

        # Обновляем оценку трения C_hat
        # C_hat_dot = alpha * delta_E * m * l^2 * |S2|^3
        C_hat_dot = self.alpha * delta_E * self.m * self.l**2 * np.abs(S2)**3
        self.C_hat += C_hat_dot * self.dt # Простое Эйлерово интегрирование

        # Ограничиваем C_hat снизу нулем, если это физически оправдано (трение не может быть отрицательным)
        # self.C_hat = max(0, self.C_hat) # Опционально

        return torque

    def get_estimate(self):
        """Возвращает текущую оценку коэффициента трения."""
        return self.C_hat 