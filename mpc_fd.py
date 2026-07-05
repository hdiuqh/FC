import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.interpolate import interp1d
from scipy.optimize import minimize


class SparseMPC:

    def __init__(self):

        # Vehicle Parameters

        self.dt = 0.05

        self.vx = 8.0

        self.L = 2.7
        # MPC Parameters

        self.N = 20

        self.interval = 5

        self.nx = 2

        self.nu = 1

        # Sparse control index

        self.sparse_index = np.arange(
            0,
            self.N + 1,
            self.interval
        )

        if self.sparse_index[-1] != self.N:

            self.sparse_index = np.append(
                self.sparse_index,
                self.N
            )

        self.ns = len(self.sparse_index)

        # Cost Weight

        self.Q = np.diag([30.0, 10.0])

        self.R = np.array([[0.5]])

        # Constraint

        self.delta_max = np.deg2rad(30)

        self.delta_min = -self.delta_max

        # Linear Vehicle Model

        self.A, self.B = self.build_model()


    # Vehicle Model

    # x=[ey,epsi]

    def build_model(self):

        A = np.array([

            [1.0,
             self.vx * self.dt],

            [0.0,
             1.0]

        ])

        B = np.array([

            [0.0],

            [self.vx * self.dt / self.L]

        ])

        return A, B


    # Prediction Matrix

    # X=Fx+GU

    def prediction_matrix(self):

        F = np.zeros(

            (self.nx * self.N,
             self.nx)

        )

        G = np.zeros(

            (self.nx * self.N,
             self.nu * self.N)

        )

        for i in range(self.N):

            A_power = np.linalg.matrix_power(
                self.A,
                i + 1
            )

            F[
                i*self.nx:(i+1)*self.nx,
                :
            ] = A_power

            for j in range(i + 1):

                G[
                    i*self.nx:(i+1)*self.nx,
                    j:(j+1)
                ] = np.linalg.matrix_power(
                    self.A,
                    i-j
                ) @ self.B

        return F, G

    def interpolation_matrix(self):

        T = np.zeros(

            (
                self.N,
                self.ns
            )

        )

        x_dense = np.arange(self.N)

        for k in range(self.ns):

            y = np.zeros(self.ns)

            y[k] = 1

            f = interp1d(

                self.sparse_index,

                y,

                kind='linear',

                fill_value="extrapolate"

            )

            T[:, k] = f(x_dense)

        return T

    # Cost Matrix

    def cost_matrix(self):

        Qbar = block_diag(

            *([self.Q] * self.N)

        )

        Rbar = block_diag(

            *([self.R] * self.N)

        )

        return Qbar, Rbar

    # Quadratic Cost

    def objective(

            self,

            u_sparse,

            x0

    ):

        F, G = self.prediction_matrix()

        T = self.interpolation_matrix()

        Qbar, Rbar = self.cost_matrix()

        u_dense = T @ u_sparse

        X = F @ x0 + G @ u_dense

        J_state = X.T @ Qbar @ X

        J_control = u_dense.T @ Rbar @ u_dense

        J = J_state + J_control

        return float(J)