
package project1




import scala.io.Source
import scalation.mathstat.{VectorD, MatrixD, PlotM}
import scalation.modeling.{TranRegression, Regression}
import scala.math.{ sqrt }
import scalation.modeling.qk


@main def TransformedRegressionSqrt(): Unit =
//   println("Hello, World!")

    
    val ox_fname = Array ("mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin")

    
    val filePath = "/mnt/c/Libs/scalation_2.0/data/auto-mpg.csv"

    val data: Array[Array[String]] = Source.fromFile(filePath)
        .getLines()
        .drop(1)
        .map(_.split(","))
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

    

    def f (u:Double): Double = sqrt(u)
    def fi(u:Double): Double = sq(u)
    val mod = new TranRegression (x, y,  ox_fname, Regression.hp, f, fi)

    mod.trainNtest ()()                                       // train and test the model
    println (mod.summary ())                                  // parameter/coefficient statistics

    banner ("Quality of Fit - based on transformed data")
    println (mod.report (mod.test0 ()._2))                    // test with transformed


    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    banner ("Validation")
    mod.validate ()()

    banner ("cross-validation")
    mod.crossValidate ()

    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    banner ("Forward Selection Test")
    val (cols1, rSq1) = mod.forwardSelAll (cross = false)                         // R^2, R^2 bar, sMAPE, R^2 cv
    val k1 = cols1.size
    val t = VectorD.range (1, k1)                                   // instance index
    new PlotM (t, rSq1.transpose, Regression.metrics, "R^2 vs n for Regression", lines = true)
    println (s"rSq = $rSq1")                                       // train and test the model
    
    banner ("Feature Importance")
    val imp1 = mod.importance (cols1.toArray, rSq1)
    for (c, r) <- imp1 do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r") 

    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    banner ("Backward Elimination Test")
    val (cols2, rSq2) = mod.backwardElimAll (cross = false)
    val k2 = cols2.size
    println (s"k = $k2")
    new PlotM (null, rSq2.transpose, Regression.metrics, s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq2")
    banner ("Feature Importance")
    val imp2 = mod.importance (cols2.toArray, rSq2)
    for (c, r) <- imp2 do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r") 

    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    println("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    banner ("Stepwise FS Test")
    val (cols3, rSq3) = mod.stepwiseSelAll (cross = false)                             // R^2, R^2 bar, sMAPE, R^2 cv

    val k3 = cols3.size
    println (s"k = $k3")
    new PlotM (null, rSq3.transpose, Regression.metrics, s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq3")
    banner ("Feature Importance")
    val imp3 = mod.importance (cols3.toArray, rSq3)
    for (c, r) <- imp3 do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r") 

