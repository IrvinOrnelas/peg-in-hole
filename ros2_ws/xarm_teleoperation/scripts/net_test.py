"""
net_test.py — Prueba de conectividad UDP/TCP entre maestro y esclavo.
Ejecutar en AMBAS computadoras para verificar la red antes del examen.

UDP:
  En PC-A (receptor):   python3 net_test.py --mode server --protocol udp
  En PC-B (emisor):     python3 net_test.py --mode client --protocol udp --ip <IP-PC-A>

TCP:
  En PC-A (receptor):   python3 net_test.py --mode server --protocol tcp
  En PC-B (emisor):     python3 net_test.py --mode client --protocol tcp --ip <IP-PC-A>
"""

import socket
import time
import argparse
import statistics

PORT = 9999
N_PACKETS = 100   # número de paquetes de prueba
RATE_HZ = 100.0
PERIOD = 1.0 / RATE_HZ


def print_stats(rtts, lost):
    total = N_PACKETS
    if rtts:
        print(f"\n{'='*50}")
        print(f"  Paquetes enviados  : {total}")
        print(f"  Paquetes perdidos  : {lost} ({100*lost/total:.1f}%)")
        print(f"  RTT mínimo         : {min(rtts):.2f} ms")
        print(f"  RTT máximo         : {max(rtts):.2f} ms")
        print(f"  RTT promedio       : {statistics.mean(rtts):.2f} ms")
        if len(rtts) > 1:
            print(f"  Desv. estándar RTT : {statistics.stdev(rtts):.2f} ms")
        else:
            print(f"  Desv. estándar RTT : 0.00 ms")
        print(f"{'='*50}")

        if statistics.mean(rtts) < 10.0 and lost / total < 0.01:
            print("  ✓ RED APTA para control de impedancia (< 10 ms, < 1% pérdida)")
        else:
            print("  ✗ RED INADECUADA — revisar conexión WiFi o usar cable")
    else:
        print("ERROR: Sin respuesta del servidor. Verificar IP, firewall y conexión.")


# =========================================================
# UDP
# =========================================================
def run_server_udp():
    """
    Servidor UDP: recibe paquetes y responde con eco inmediato.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', PORT))
    print(f"[SERVIDOR UDP] Esperando en puerto {PORT}...")

    count = 0
    while count < N_PACKETS:
        data, addr = sock.recvfrom(128)
        sock.sendto(data, addr)
        count += 1

    print(f"[SERVIDOR UDP] {count} paquetes procesados. ¡Conectividad OK!")
    sock.close()


def run_client_udp(server_ip):
    """
    Cliente UDP: envía paquetes de prueba y mide RTT.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)

    rtts = []
    lost = 0

    print(f"[CLIENTE UDP] Enviando {N_PACKETS} paquetes a {server_ip}:{PORT}...")
    for i in range(N_PACKETS):
        payload = f"PING_{i:04d}_{time.time():.6f}".encode()
        t0 = time.perf_counter()
        sock.sendto(payload, (server_ip, PORT))

        try:
            resp, _ = sock.recvfrom(128)
            if resp == payload:
                rtt_ms = (time.perf_counter() - t0) * 1000.0
                rtts.append(rtt_ms)
            else:
                lost += 1
        except socket.timeout:
            lost += 1

        time.sleep(PERIOD)

    sock.close()
    print_stats(rtts, lost)


# =========================================================
# TCP
# =========================================================
def recv_exact(sock, nbytes):
    """
    Recibe exactamente nbytes desde un socket TCP.
    Devuelve bytes o None si la conexión se cerró.
    """
    data = b""
    while len(data) < nbytes:
        chunk = sock.recv(nbytes - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def run_server_tcp():
    """
    Servidor TCP: acepta una conexión, recibe paquetes y responde con eco.
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('', PORT))
    server_sock.listen(1)

    print(f"[SERVIDOR TCP] Esperando conexión en puerto {PORT}...")
    conn, addr = server_sock.accept()
    print(f"[SERVIDOR TCP] Cliente conectado desde {addr}")

    count = 0
    try:
        while count < N_PACKETS:
            data = conn.recv(128)
            if not data:
                break
            conn.sendall(data)
            count += 1
    finally:
        print(f"[SERVIDOR TCP] {count} paquetes procesados.")
        conn.close()
        server_sock.close()

    if count == N_PACKETS:
        print("[SERVIDOR TCP] ¡Conectividad OK!")
    else:
        print("[SERVIDOR TCP] Conexión cerrada antes de completar la prueba.")


def run_client_tcp(server_ip):
    """
    Cliente TCP: se conecta al servidor, envía paquetes de prueba y mide RTT.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)

    print(f"[CLIENTE TCP] Conectando a {server_ip}:{PORT}...")
    sock.connect((server_ip, PORT))
    sock.settimeout(1.0)

    rtts = []
    lost = 0

    print(f"[CLIENTE TCP] Enviando {N_PACKETS} paquetes...")
    for i in range(N_PACKETS):
        payload = f"PING_{i:04d}_{time.time():.6f}".encode()
        t0 = time.perf_counter()

        try:
            sock.sendall(payload)
            resp = recv_exact(sock, len(payload))

            if resp == payload:
                rtt_ms = (time.perf_counter() - t0) * 1000.0
                rtts.append(rtt_ms)
            else:
                lost += 1

        except socket.timeout:
            lost += 1
        except (BrokenPipeError, ConnectionResetError):
            lost += 1
            break

        time.sleep(PERIOD)

    sock.close()
    print_stats(rtts, lost)


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de red UDP/TCP")
    parser.add_argument(
        "--mode",
        choices=["server", "client"],
        required=True,
        help="Modo de ejecución"
    )
    parser.add_argument(
        "--protocol",
        choices=["udp", "tcp"],
        default="udp",
        help="Protocolo de transporte"
    )
    parser.add_argument(
        "--ip",
        default="127.0.0.1",
        help="IP del servidor (solo en modo cliente)"
    )

    args = parser.parse_args()

    if args.protocol == "udp":
        if args.mode == "server":
            run_server_udp()
        else:
            run_client_udp(args.ip)
    else:
        if args.mode == "server":
            run_server_tcp()
        else:
            run_client_tcp(args.ip)