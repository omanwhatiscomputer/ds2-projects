package project1.EE
import scalation.mathstat.{VectorD, MatrixD}
import scala.math.{log, exp, pow}
import scalation.~^

def banner (str: String): Unit =
    val len = str.size + 4
    println ("-" * len)
    println ("| " + str + " |")
    println ("-" * len)
end banner

def minCol(m: MatrixD): VectorD = VectorD((0 until m.dim2).map(j => m.col(j).min))
def maxCol(m: MatrixD): VectorD = VectorD((0 until m.dim2).map(j => m.col(j).max))

def box_cox (y: Double): Double =
    val lambda = 0.5
    if lambda == 0.0 then log (y)
    else (y ~^ lambda - 1.0) / lambda
end box_cox

def cox_box (z: Double): Double =
    val lambda = 0.5
    if lambda == 0.0 then exp (z)
    else (lambda * z + 1.0) ~^ (1.0 / lambda)
end cox_box

def yJ(y: Double): Double =
    val lambda = 0.5
    if y >= 0 then
        if lambda != 0.0 then
        (pow(y + 1.0, lambda) - 1.0) / lambda
        else
        log(y + 1.0)
    else // y < 0
        if lambda != 2.0 then
        -(pow(-y + 1.0, 2.0 - lambda) - 1.0) / (2.0 - lambda)
        else
        -log(-y + 1.0)
end yJ

def iYJ(z: Double): Double =
    val lambda = 0.5
    if z >= 0 then
        if lambda != 0.0 then
        pow(lambda * z + 1.0, 1.0 / lambda) - 1.0
        else
        exp(z) - 1.0
    else // z < 0
        if lambda != 2.0 then
        1.0 - pow(-(2.0 - lambda) * z + 1.0, 1.0 / (2.0 - lambda))
        else
        1.0 - exp(-z)
end iYJ

inline def sq (x: Double): Double = x * x