import sys
import os
import time
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from classes.network.SlaveNetServer import SlaveNetServer


def main():
    parser = argparse.ArgumentParser(description="Slave UDP dummy")
    parser.add_argument(
        "--master-ip",
        default="127.0.0.1",
        help="IP address of the master computer"
    )
    args = parser.parse_args()

    server = SlaveNetServer(master_ip=args.master_ip)

    print(f"Slave dummy corriendo...")
    print(f"Sending responses to master IP: {args.master_ip}")

    while True:
        print("received p_des:", server.p_des)

        server.send_contact_data(
            F_contact=np.array([1.0, -0.5], dtype=np.float64),
            in_contact=True,
            contact_state="CONTACT"
        )

        time.sleep(0.1)


if __name__ == "__main__":
    main()