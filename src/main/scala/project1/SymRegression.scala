
package project1




import scala.io.Source
import scalation.mathstat.{VectorD, MatrixD, PlotM}
import scalation.modeling.{SymbolicRegression, Regression, SelectionTech}
import scalation.modeling.qk


@main def SymRegression(): Unit =
//   println("Hello, World!")

    
    val filePath = "/mnt/c/Libs/scalation_2.0/data/auto-mpg.csv"

    val data: Array[Array[String]] = Source.fromFile(filePath)
        .getLines()
        .drop(1)
        .map(_.split(",").drop(1))
        .filter(row => row.forall(_.nonEmpty))
        .toArray

    // Extract y
    val y = VectorD(data.map(_(0).toDouble))

    // Extract x
    val xRows = data.map(row => row.drop(1).map(_.toDouble))
    val x = MatrixD(xRows.map(row => VectorD(row)).toIndexedSeq)

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    // Extract y
    // val y = VectorD(data.map(_(0).toDouble))

    // Extract x
    // val xRows = data.map(row => row.drop(1).map(_.toDouble))
    // val xRaw = MatrixD(xRows.map(row => VectorD(row)).toIndexedSeq)

    // val xMin = minCol(xRaw)
    // val xMax = maxCol(xRaw)

    // val x = MatrixD((0 until xRaw.dim).map(i => {
    //     val row = xRaw(i)
    //     VectorD(row.zipWithIndex.map { case (v, j) =>
    //         if (xMax(j) != xMin(j)) then (v - xMin(j)) / (xMax(j) - xMin(j)) else 0.0
    //     })
    // }).toIndexedSeq)

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    val mod = SymbolicRegression (x, y, null, scala.collection.mutable.Set(-2.0, -1.0, 0.5, 2.0))    // add, intercept, cross-terms and given powers
    mod.trainNtest ()()                                                 // train and test the model
    println (mod.summary ())                                            // parameter/coefficient statistics

    for tech <- SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                     // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Regression.metrics, s"R^2 vs n for Quadratic X Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for