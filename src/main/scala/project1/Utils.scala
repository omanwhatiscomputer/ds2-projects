package project1
import scalation.mathstat.{VectorD, MatrixD}

def banner (str: String): Unit =
    val len = str.size + 4
    println ("-" * len)
    println ("| " + str + " |")
    println ("-" * len)
end banner

def minCol(m: MatrixD): VectorD = VectorD((0 until m.dim2).map(j => m.col(j).min))
def maxCol(m: MatrixD): VectorD = VectorD((0 until m.dim2).map(j => m.col(j).max))