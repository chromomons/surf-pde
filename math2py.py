from sympy.parsing.mathematica import mathematica

expr = "(x^5*(-3*y^2 - 30*y*z + 2*z^2) + 2*x^3*(3*y^4 + 15*y^3*z + 4*y^2*z^2 + 30*y*z^3 + z^4) + 15*x*(4*y^5*z + 4*y^3*z^3 - 15*z^6))/(x^2 + y^2 + z^2)^2, (3*x^6*(y + 10*z) - 15*y*z^3*(2*y^3 + 2*y*z^2 + 15*z^3) + x^4*(-6*y^3 - 30*y^2*z + 5*y*z^2 + 45*z^3) + x^2*z*(-60*y^4 + 2*y^3*z + 15*y^2*z^2 + 2*y*z^3 + 15*z^4))/(x^2 + y^2 + z^2)^2, (z*(-2*x^6 - x^4*(13*y^2 + 105*y*z + 2*z^2) + 15*y^2*z*(2*y^3 + 2*y*z^2 + 15*z^3) - x^2*(2*y^4 + 75*y^3*z + 2*y^2*z^2 + 15*y*z^3 - 225*z^4)))/(x^2 + y^2 + z^2)^2"

exprs = expr.split(", ")

for expi in exprs:
    print(mathematica(expi))
    print("")
