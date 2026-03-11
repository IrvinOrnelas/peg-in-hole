import sys
import os
import time
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from classes.network.MasterNetClient import MasterNetClient


def main():
    parser = argparse.ArgumentParser(description="Master UDP dummy")
    parser.add_argument(
        "--slave-ip",
        default="127.0.0.1",
        help="IP address of the slave computer"
    )
    args = parser.parse_args()

    client = MasterNetClient(slave_ip=args.slave_ip)

    print(f"Master dummy corriendo...")
    print(f"Sending targets to slave IP: {args.slave_ip}")

    while True:
        client.send_target(np.array([0.5, 0.1], dtype=np.float64))
        print(client.F_contact, client.in_contact, client.contact_state)
        time.sleep(0.1)


if __name__ == "__main__":
    main()