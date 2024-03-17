import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import scipy.optimize as opt
from dataclasses import dataclass

from utilities import generate_random_joint_positions


class InverseKinematicsSLSQP:
    @dataclass
    class Result:
        q: np.array
        success: bool
        error: float
        final_pose: sm.SE3
        iterations: int
    
    def __init__(self, robot_model: rtb.Robot):
        self._robot_model: rtb.Robot = robot_model
        self._error_last: float = 0
        self._q0: np.array
        self._target_pose: sm.SE3
        self._final_pose: sm.SE3
        self._q0_weight: float
        self._q_weight: float

    def _angular_distance(self, T1: sm.SE3, T2: sm.SE3):
        return np.trace(np.eye(3) - np.dot(T1.R, T2.R.T))

    def _cost_fun(self, q: np.array):
        q_diff = q - self._q0
        return self._q0_weight*np.dot(q_diff, q_diff) + np.dot(q, q)
    
    def _cost_fun_derivative(self, q: np.array):
        q_diff = q - self._q0
        return self._q0_weight*2*q_diff + 2*self._q_center_weight*q

    def _pose_error(self, q: np.array):
        Trobot = self._robot_model.fkine(q, end=self._robot_model.ee_links[0])
        pos_diff = self._target_pose.t - Trobot.t
        pos_error = np.dot(pos_diff, pos_diff)
        ang_error = self._angular_distance(self._target_pose, Trobot)
        error = pos_error + ang_error
        self._error_last = error
        self._final_pose = Trobot
        return error
    
    # TODO: Compute the Jacobian of the pose error

    def solve(
        self, 
        target_pose: sm.SE3, 
        q0: np.array, 
        q_min: np.array, 
        q_max: np.array, 
        max_trials: int=100, 
        tolerance: float=0.001,
        q0_weight: float=5,
        q_center_weight: float=1
    ) -> Result:
        '''
        Solves the inverse kinematics problem using the Sequential Least Squares Programming (SLSQP) algorithm. The target pose is encoded in the constraints of the optimization problem. The cost function is a weighted sum of the initial joint position and the joint position itself. The initial joint position is weighted with the q0Weight and the joint position itself is weighted with the qCenterWeight. Multiple optmizations are run until a solution is found or max_trials is reached. The first iteration uses the initial guess q0, the following iterations use random initial guesses to reach a feasible region. 

        :param target_pose: The desired end-effector pose.
        :type target_pose: sm.SE3
        :param q0: The starting point and also initial guess for the joint positions.
        :type q0: np.array
        :param q_min: The lower joint limits.
        :type q_min: np.array
        :param q_max: The upper joint limits.
        :type q_max: np.array
        :param max_trials: The maximum number of trials
        :type max_trials: int, optional
        :param tolerance: The tolerance for the optimization algorithm convergence.
        :type tolerance: float, optional
        :param q0_weight: This weight drives the solution to joint positions that are close to q0.
        :type q0_weight: float, optional
        :param q_center_weight: This weight drives the solution to centered joint positions.
        '''
        
        self._target_pose = target_pose
        self._q0 = q0
        self._q0_weight = q0_weight
        self._q_center_weight = q_center_weight
        bounds = opt.Bounds(q_min, q_max)
        constraints = [{'type': 'eq', 'fun': self._pose_error}]

        for i in range(max_trials):
            initial_guess = q0 if i == 0 else generate_random_joint_positions(q_min, q_max)
            result = opt.minimize(
                fun=self._cost_fun, 
                x0=initial_guess, 
                jac=self._cost_fun_derivative, 
                bounds=bounds, 
                constraints=constraints, 
                tol=tolerance, 
                options={'maxiter': 300})
            if result.success:
                break
        iterations = i + 1
        
        return self.Result(result.x, result.success, self._error_last, self._final_pose, iterations)





###############################
# TEST
###############################
if __name__ == '__main__':
    import time
    from utilities import generate_random_joint_positions

    # Create robot model (make use of the URDF reading capabilities of rtb to load an arbitrary robot model)
    robot = rtb.models.KinovaGen3()
    ik_solver = InverseKinematicsSLSQP(robot)
    
    # Joint limits
    INF_LIMIT = 180
    Q_MAX = np.deg2rad(np.array([INF_LIMIT, 128.9, INF_LIMIT, 147.8, INF_LIMIT, 120.3, INF_LIMIT]))
    Q_MIN = -Q_MAX

    # Generate random target pose 
    qf = generate_random_joint_positions(Q_MIN, Q_MAX)
    target_pose = robot.fkine(qf, end=robot.ee_links[0])

    # Solve inverse kinematics
    q0 = np.zeros_like(qf)
    start = time.time()
    result = ik_solver.solve(target_pose, q0, Q_MIN, Q_MAX)
    duration = time.time() - start

    # Print results
    print(f'Success: {result.success}')
    print(f'Duration: {duration}')
    print(f'Iterations: {result.iterations}')
    print(f'Error: {result.error}')
    print(f'Final pose:\n {result.final_pose}')
    print(f'Final joint positions: {result.q}')
