
package project1




import scala.io.Source
import scalation.mathstat.{VectorD, MatrixD}
import scalation.modeling.{TranRegression, Regression}
import scala.math.{ log1p, sqrt, expm1}

@main def TransformedRegressionYJ(): Unit =
//   println("Hello, World!")

    
    val filePath = "/mnt/c/Libs/scalation_2.0/data/auto-mpg.csv"

    val data: Array[Array[String]] = Source.fromFile(filePath)
        .getLines()
        .drop(1)
        .map(_.split(",").drop(1))
        .filter(row => row.forall(_.nonEmpty))
        .toArray

    // Extract y
    // val y = VectorD(data.map(_(0).toDouble))

    // Extract x
    // val xRows = data.map(row => row.drop(1).map(_.toDouble))
    // val x = MatrixD(xRows.map(row => VectorD(row)).toIndexedSeq)

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    // Extract y
    val y = VectorD(data.map(_(0).toDouble))

    // Extract x
    val xRows = data.map(row => row.drop(1).map(_.toDouble))
    val xRaw = MatrixD(xRows.map(row => VectorD(row)).toIndexedSeq)

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

    val t = (sqrt,  log1p,   box_cox, yJ)
    val ti = (sq, expm1, cox_box, iYJ)

    def f (u:Double): Double = t._1(t._2(t._3(t._4(u))))
    def fi(u:Double): Double = ti._4(ti._3(ti._2(ti._1(u))))
    val mod = new TranRegression (x, y,  null, Regression.hp, f, fi)

    mod.trainNtest ()()                                       // train and test the model
    println (mod.summary ())                                  // parameter/coefficient statistics

    banner ("Quality of Fit - based on transformed data")
    println (mod.report (mod.test0 ()._2))                    // test with transformed



