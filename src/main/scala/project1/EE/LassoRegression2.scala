

package project1.EE




import scala.io.Source

import scalation.mathstat.{VectorD, MatrixD, PlotM}
import scalation.modeling.{LassoRegression, Regression}
import scalation.modeling.qk


@main def LassoRegression2(): Unit =
//   println("Hello, World!")

    val ox_fname = Array ("X1","X2","X3","X4","X5","X6","X7","X8","Y1","Y2")
    
    val filePath = "/mnt/c/Libs/scalation_2.0/data/energy-efficiency.csv"

    val data: Array[Array[String]] = Source.fromFile(filePath)
        .getLines()
        .drop(1)                           // skip header
        .map(_.split(","))
        .filter(row => row.forall(_.nonEmpty))
        .toArray

    // Extract y (Y1 is column 8, index 8 because 0-based)
    val y = VectorD(data.map(_(8).toDouble))

    // Extract X (first 8 columns: indices 0 to 7)
    val xRows = data.map(row => row.take(8).map(_.toDouble))
    val x = MatrixD(xRows.map(row => VectorD(row)).toIndexedSeq)

    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    // Print first 5 y values and x rows for verification
    // println("y (displacement): " + y(0 until 5))
    // println("x (features):")
    // println(x(0 until 5))

    val mod = new LassoRegression (x, y, ox_fname)                          // create a simple regression model
    mod.trainNtest ()()
    println (mod.summary ())                                  // parameter/coefficient statistics

    
    
    // for tech <- SelectionTech.values do 
    //     banner (s"Feature Selection Technique: $tech")
    //     val (cols, rSq) = mod.selectFeatures (tech)                     // R^2, R^2 bar, R^2 cv
    //     val k = cols.size
    //     println (s"k = $k, n = ${x.dim2}")
    //     new PlotM (null, rSq.transpose, Regression.metrics, s"R^2 vs n for Quadratic X Regression with $tech", lines = true)
    //     println (s"$tech: rSq = $rSq")
    // end for     

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


