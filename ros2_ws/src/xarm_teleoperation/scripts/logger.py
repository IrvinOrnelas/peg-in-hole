#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import csv
import time
import numpy as np

class TeleopLogger(Node):
    def __init__(self):
        super().__init__('teleop_logger')

        # Variables para almacenar los estados más recientes
        self.q_master = None
        self.q_slave = None
        self.start_time = time.time()

        # Suscriptores a los tópicos REALES de ambos robots
        self.create_subscription(JointState, '/leader/joint_states', self.master_cb, 10)
        self.create_subscription(JointState, '/follower/joint_states', self.slave_cb, 10)

        # Configuración del archivo CSV
        self.csv_file = open('ensayo_seguimiento.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Encabezados exactos que usa nuestro script de graficación
        encabezados = ['tiempo', 
                       'qm1', 'qm2', 'qm3', 'qm4', 'qm5', 'qm6', 
                       'qs1', 'qs2', 'qs3', 'qs4', 'qs5', 'qs6']
        self.csv_writer.writerow(encabezados)

        # Timer para guardar datos a 50 Hz (0.02 segundos)
        self.timer = self.create_timer(0.02, self.log_data)
        self.get_logger().info('Logger iniciado. Esperando datos de ambos robots...')

    def master_cb(self, msg):
        if len(msg.position) >= 6:
            self.q_master = np.array(msg.position[:6])

    def slave_cb(self, msg):
        if len(msg.position) >= 6:
            self.q_slave = np.array(msg.position[:6])

    def log_data(self):
        # Solo guardamos si ya estamos recibiendo datos de AMBOS robots
        if self.q_master is not None and self.q_slave is not None:
            current_time = time.time() - self.start_time
            
            # Unimos el tiempo + 6 valores del master + 6 valores del slave
            fila = [current_time] + self.q_master.tolist() + self.q_slave.tolist()
            self.csv_writer.writerow(fila)

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info('Archivo CSV cerrado y guardado con éxito.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TeleopLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()