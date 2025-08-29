
package project1




import scala.io.Source
import scalation.mathstat.{VectorD, MatrixD}
import scalation.modeling.{RidgeRegression, Regression}



def minCol(m: MatrixD): VectorD = VectorD((0 until m.dim2).map(j => m.col(j).min))
def maxCol(m: MatrixD): VectorD = VectorD((0 until m.dim2).map(j => m.col(j).max))

def banner (str: String): Unit =
    val len = str.size + 4
    println ("-" * len)
    println ("| " + str + " |")
    println ("-" * len)
end banner

@main def RidgeRegression(): Unit =
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

    banner ("Regression")
    val ox  = VectorD.one (y.dim) +^: x                                // prepend a column of all 1's
    val reg = new Regression (ox, y)                                   // create a Regression model
    reg.trainNtest ()()                                                // train and test the model

    banner ("RidgeRegression")
    val mu_x = x.mean                                                  // column-wise mean of x
    val mu_y = y.mean                                                  // mean of y
    val x_c  = x - mu_x                                                // centered x (column-wise)
    val y_c  = y - mu_y                                                // centered y
    val mod  = new RidgeRegression (x_c, y_c)                          // create a Ridge Regression model
    mod.trainNtest ()()                                                // train and test the model

    

    banner ("Compare Summaries")
    println (reg.summary ())
    println (mod.summary ())