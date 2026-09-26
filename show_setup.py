"""[O] Show who implements which algorithm, and the machine / protocol to report.

Usage:
    python show_setup.py
"""
import config  # first import: pins BLAS threads before NumPy loads

from common import sysinfo


def main():
    print(f"CEG5201 CA2 - Group {config.GROUP_ID}\n")
    print(f"{'ID':<4}{'Name':<24}{'Matric':<12}{'n':<3}{'Algorithm':<22}File")
    for member_id, name, matric in config.MEMBERS:
        n = config.algorithm_number(member_id)
        print(f"{member_id:<4}{name:<24}{matric:<12}{n:<3}{config.ALGORITHM_TITLES[n]:<22}"
              f"algorithms/{config.ALGORITHMS[n]}.py")
    print()
    print(sysinfo.summary())


if __name__ == "__main__":
    main()
