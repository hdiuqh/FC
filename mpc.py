import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from mpc_fd import SparseMPC

def constraint(self):

    bound = []

    for _ in range(self.ns):

        bound.append(

            (

                self.delta_min,

                self.delta_max

            )

        )

    return bound

    # Solve Sparse MPC

def solve(self, x0):

    u0 = np.zeros(self.ns)

    result = minimize(

        self.objective,

        u0,

        args=(x0),

        method='SLSQP',

        bounds=self.constraint(),

        options={

            'maxiter':100,

            'disp':False

        }

    )

    u_sparse = result.x

    return u_sparse

    # Sparse → Dense

def dense_control(

        self,

        u_sparse

    ):

    T = self.interpolation_matrix()

    u_dense = T @ u_sparse

    return u_dense

    # Complete MPC

def control(

        self,

        x0

    ):

    u_sparse = self.solve(x0)

    u_dense = self.dense_control(

        u_sparse

    )

    return u_sparse,u_dense

    # Visualization

def plot(

        self,

        u_sparse,

        u_dense

    ):

    plt.figure(

        figsize=(10,5)

    )

    plt.plot(

        np.arange(self.N),

        u_dense,

        linewidth=2,

        label="Dense MPC"

        )

    plt.scatter(

        self.sparse_index,

        u_sparse,

        s=70,

        marker='o',

        label="Sparse Key Points"

    )

    plt.grid(True)

    plt.xlabel(

        "Prediction Step"

        )

    plt.ylabel(

        "Steering (rad)"

    )

    plt.legend()

    plt.tight_layout()

    plt.show()


if __name__=="__main__":

    controller=SparseMPC()

    x0=np.array([

        0.25,

        np.deg2rad(5)

    ])

    u_sparse,u_dense=\

    controller.control(x0)

    print(

        "Sparse Control Sequence"

    )

    print(u_sparse)

    print(

        "Dense Control Sequence"

    )

    print(u_dense)

    controller.plot(

        u_sparse,

        u_dense

    )