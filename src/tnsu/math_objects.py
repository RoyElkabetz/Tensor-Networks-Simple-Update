import numpy as np


def spin_operators(spin: float = 0.5):
    """
    (S_x)_{ab} = 1/2 (\\delta_{a,b+1} + \\delta_{a+1,b}) \\sqrt{(s+1)(a+b-1)-ab}
    (S_y)_{ab} = i/2 (\\delta_{a,b+1} - \\delta_{a+1,b}) \\sqrt{(s+1)(a+b-1)-ab}
    (S_z)_{ab} =     (s + 1 - a) \\delta_{a,b}
    :param spin: a half integer for spin type
    :retrun: the three spin operators sx, sy, sz
    """
    n = int(2 * spin + 1)
    assert spin > 0, f"the spin should be a positive half integer, instead got {spin}."
    assert (n - (2 * spin + 1)) == 0, f"spin should be a half integer (i.e., 0.5, 1, 1.5, ...), instead got {spin}."
    sx = np.zeros((n, n), dtype=complex)
    sy = np.zeros((n, n), dtype=complex)
    sz = np.zeros((n, n), dtype=complex)

    for i in range(n):
        for j in range(n):
            a = i + 1
            b = j + 1
            sx[i, j] = 0.5 * (int(a == b + 1) + int(a + 1 == b)) * np.sqrt((spin + 1) * (a + b - 1) - a * b)
            sy[i, j] = 0.5j * (int(a == b + 1) - int(a + 1 == b)) * np.sqrt((spin + 1) * (a + b - 1) - a * b)
            sz[i, j] = (spin + 1 - a) * int(a == b)

    return sx, sy, sz
