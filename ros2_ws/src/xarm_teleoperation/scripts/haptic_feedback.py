#!/usr/bin/env python3

import csv
import os
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
)
from sensor_msgs.msg import JointState
from xarm_msgs.srv import MoveJoint
from std_msgs.msg import Float64
from xarm_msgs.srv import MoveJoint, SetInt16
from xarm_teleoperation.kinematics import position_jacobian


class HapticFeedbackNode(Node):

    def __init__(self):

        super().__init__('haptic_feedback_node')

        self.beta = 0.7
        self.K_virtual = 50.0
        self.umbral_min = 1.0
        self.umbral_max = 10.0

        self.q_master = np.zeros(6)
        self.force_z = 0.0
    
    
        self.master_mode = 2
        
        self.in_contact = False
        self.kick_counter = 0
        self.duration_kick = 100
        self.force_kick_extra = 25.0

        self.create_subscription(JointState,"/leader/joint_states" , self.joint_callback, 10)
        self.create_subscription(Float64,   "force_sensor/force", self.force_callback, 10)

        self.leader_client = self.create_client(MoveJoint, '/leader/set_servo_angle')
        
        self.mode_client = self.create_client(SetInt16, '/leader/set_mode')
        self.state_client = self.create_client(SetInt16, '/leader/set_state')
        
        while not self.leader_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Esperando servicio del leader...')

        self.waiting_for_response = False

        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info("Haptic feedback node started")

    def joint_callback(self, msg):

        if len(msg.position) >= 6:
            self.q_master = np.array(msg.position[:6])

    def force_callback(self, msg):
        if abs(msg.data) > self.umbral_min and abs(msg.data) < self.umbral_max:  #
            self.force_z = msg.data
        else:
            self.force_z = 0.0
            
    def switch_master_mode(self, mode):
        if self.master_mode == mode:
            return # Si ya estamos en ese modo, no hacemos nada para evitar spam a la red

        self.get_logger().info(f"Cambiando Master a Modo {mode}...")
        
        # 1. Cambiamos el modo (1 = Servo/Rigido, 2 = Teaching/Libre)
        req_mode = SetInt16.Request()
        req_mode.data = mode
        self.mode_client.call_async(req_mode)
        
        # 2. Obligatorio: cambiar el estado a 0 (Sport) para que el modo surta efecto
        req_state = SetInt16.Request()
        req_state.data = 0
        self.state_client.call_async(req_state)
        
        self.master_mode = mode


    def control_loop(self):
        
        if self.q_master is None or self.waiting_for_response:
            return
        
        there_contact = (self.force_z != 0.0)
        
        if there_contact and not self.in_contact:
            # EVENTO: Acabamos de chocar contra la pared
            self.switch_master_mode(0)
            self.switch_master_mode(0)
            self.in_contact = True
            
            # Activamos el temporizador de la fuerza artificial
            self.kick_counter = self.duration_kick

        elif not there_contact and self.in_contact:
            # EVENTO: Acabamos de liberar la pared
            self.switch_master_mode(2)
            self.switch_master_mode(2)
            self.in_contact = False

        # Si no estamos en contacto, no hay nada más que calcular
        if not self.in_contact:
            return

        q = self.q_master
        J = position_jacobian(q) 
        
        
        # Tomamos la fuerza real medida
        apply_force = self.force_z
        
        # Inyectamos el "empujón" artificial si el contador está activo
        if self.kick_counter > 0:
            # Determinamos la dirección del empujón basándonos en la fuerza leída
            # (Asumimos que queremos sumar fuerza en la misma dirección del impacto para alejar la mano)
            sign = 1 if self.force_z > 0 else -1
            apply_force += (self.force_kick_extra * sign)
            
            self.kick_counter -= 1 # Reducimos el temporizador
        
        # Asumiendo que tu position_jacobian devuelve una matriz 3x6
        F = np.array([0.0, 0.0, self.force_z])
        
        # tau_m,fb = J_m^T * beta * F_ext
        tau = J.T @ (self.beta * F)
        
        # dq = tau / K_virtual (Admitancia para simular el resorte)
        dq = tau / self.K_virtual
        
        # Sumamos el delta para empujar la mano del operador en la dirección de la fuerza
        q_cmd = q + dq
        
        print(f"Force: {self.force_z:.2f} N, Tau: {tau}, dq: {dq}, Q:{q_cmd}")
        
        if self.kick_counter > 0:
            print(f"¡KICK ACTIVO! F_total: {apply_force:.2f} N")
        else:
            print(f"F_normal: {self.force_z:.2f} N, Tau: {np.round(tau,2)}")

        req = MoveJoint.Request()
        req.angles = q_cmd.tolist()
        req.speed = 0.0
        req.acc = 0.0
        req.mvtime = 0.0
        
        self.waiting_for_response = True
        future = self.leader_client.call_async(req)
        future.add_done_callback(self.service_done_callback)
        
    def service_done_callback(self, future):
            self.waiting_for_response = False


def main(args=None):

    rclpy.init(args=args)

    node = HapticFeedbackNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()