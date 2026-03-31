import cover_float.testgen.B1 as B1
import cover_float.testgen.B3 as B3
import cover_float.testgen.B4 as B4
import cover_float.testgen.B9 as B9
import cover_float.testgen.B10 as B10
import cover_float.testgen.B12 as B12
import cover_float.testgen.B14 as B14

try:
    import cover_float.testgen.B11 as B11
except ImportError:
    B11 = None

try:
    import cover_float.testgen.B15 as B15
except ImportError:
    B15 = None

__all__ = ["B1", "B3", "B4", "B9", "B10", "B12", "B14"]
if B11:
    __all__.append("B11")
if B15:
    __all__.append("B15")
