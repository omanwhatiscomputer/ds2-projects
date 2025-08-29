package project1




import scala.io.Source
import scalation.mathstat.{VectorD, MatrixD, Plot}
import scalation.modeling.SimpleRegression


def minCol(m: MatrixD): VectorD = VectorD((0 until m.dim2).map(j => m.col(j).min))
def maxCol(m: MatrixD): VectorD = VectorD((0 until m.dim2).map(j => m.col(j).max))

@main def SimpleRegression(): Unit =
//   println("Hello, World!")

    // The path to your file
    val filePath = "/mnt/c/Libs/scalation_2.0/data/auto-mpg.csv"

    val data: Array[Array[String]] = Source.fromFile(filePath)
        .getLines()
        .drop(1)
        .map(_.split(",").drop(1))
        .filter(row => row.forall(_.nonEmpty))
        .toArray

    // // Extract y (first column) and convert to VectorD
    // val y = VectorD(data.map(_(0).toDouble))

    // // Extract x (remaining columns) and convert to MatrixD
    // val xRows = data.map(row => row.drop(1).map(_.toDouble))
    // val x = MatrixD(xRows.map(row => VectorD(row)).toIndexedSeq)

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    // Extract y (first column) and convert to VectorD
    val y = VectorD(data.map(_(0).toDouble))

    // Extract x (remaining columns) and convert to MatrixD
    val xRows = data.map(row => row.drop(1).map(_.toDouble))
    val xRaw = MatrixD(xRows.map(row => VectorD(row)).toIndexedSeq)

    // Normalize x (min-max scaling for each column)
    val xMin = minCol(xRaw)
    val xMax = maxCol(xRaw)

    val x = MatrixD((0 until xRaw.dim).map(i => {
        val row = xRaw(i)
        VectorD(row.zipWithIndex.map { case (v, j) =>
            if (xMax(j) != xMin(j)) then (v - xMin(j)) / (xMax(j) - xMin(j)) else 0.0
        })
    }).toIndexedSeq)

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    // Print first 5 y values and x rows for verification
    println("y (displacement): " + y(0 until 5))
    println("x (features):")
    println(x(0 until 5))

    val mod = new SimpleRegression (x, y)                          // create a simple regression model
    mod.trainNtest ()()                                            // train and test the model

    val yp = mod.predict (x)
    new Plot (x(0 until x.dim, 1), y, yp, "plot y and yp vs. x", lines = true)