import math

def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b

def next_power_of_2(n: int) -> int:
    return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))

def get_powers_of_2(lo: int, hi: int) -> list[int]:
    out, v = [], lo
    while v <= hi:
        out.append(v); v *= 2
    return out

