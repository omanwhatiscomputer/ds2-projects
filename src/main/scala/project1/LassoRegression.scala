

package project1




import scala.io.Source

import scalation.mathstat.{VectorD, MatrixD, PlotM}
import scalation.modeling.{LassoRegression, Regression}
import scalation.modeling.qk


@main def LassoRegression(): Unit =
//   println("Hello, World!")

    // The path to your file
    val filePath = "/mnt/c/Libs/scalation_2.0/data/auto-mpg.csv"

    val data: Array[Array[String]] = Source.fromFile(filePath)
        .getLines()
        .drop(1)
        .map(_.split(",").drop(1))
        .filter(row => row.forall(_.nonEmpty))
        .toArray

    // // Extract y
    // val y = VectorD(data.map(_(0).toDouble))

    // // Extract x
    // val xRows = data.map(row => row.drop(1).map(_.toDouble))
    // val x = MatrixD(xRows.map(row => VectorD(row)).toIndexedSeq)

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    // Extract y
    val y = VectorD(data.map(_(0).toDouble))

    // Extract x
    val xRows = data.map(row => row.drop(1).map(_.toDouble))
    val xRaw = MatrixD(xRows.map(row => VectorD(row)).toIndexedSeq)

    // Normalize x
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
    // println("y (displacement): " + y(0 until 5))
    // println("x (features):")
    // println(x(0 until 5))

    banner ("LassoRegression")
    val mod = new LassoRegression (x, y, null)                    // create a Lasso regression model
    mod.trainNtest ()()                                            // train and test the model
    println (mod.summary ())                                       // parameter/coefficient statistics

    banner ("Forward Selection Test")
    val (cols, rSq) = mod.forwardSelAll ()                         // R^2, R^2 bar, sMAPE, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.transpose, Regression.metrics, "R^2 vs n for LassoRegression", lines = true)
    println (s"rSq = $rSq")