

package project2

import scala.io.Source
import scalation.mathstat.{VectorD, MatrixD}
// import scalation.modeling.qk

// import scala.math.max
// import scala.runtime.ScalaRunTime.stringOf

// import scalation.mathstat._

import scalation.modeling.ActivationFun
import scalation.modeling.neuralnet.NeuralNet_2L
// import scalation.modeling.Initializer
import scalation.modeling.neuralnet.Optimizer

@main def NeuralNetwork2LAutoMPG(): Unit =
    // val ox_fname = Array ("mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin")

    
    val filePath = "C:/Libs/scalation_2.0/data/auto-mpg.csv"

    val data: Array[Array[String]] = Source.fromFile(filePath)
        .getLines()
        .drop(1)
        .map(_.split(","))
        .filter(row => row.forall(_.nonEmpty))
        .toArray

    

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


    // Extract y
    // val yv = VectorD(data.map(_(0).toDouble))   // 1D target vector
    
    // var y = new MatrixD(yv.dim, 1)
    // for (i <- 0 until yv.dim) y(i, 0) = yv(i)
    val yv: VectorD = VectorD(data.map(_(0).toDouble).toIndexedSeq)

    // Min/Max without implicits or while/if blocks
    val yMin: Double =
        (0 until yv.dim).foldLeft(Double.PositiveInfinity)((m, i) => Math.min(m, yv(i)))
    val yMax: Double =
        (0 until yv.dim).foldLeft(Double.NegativeInfinity)((m, i) => Math.max(m, yv(i)))

    // Scale y -> [0,1] as VectorD (no braces)
    val yScaledV: VectorD =
        val range = yMax - yMin
        if range == 0.0 then VectorD.fill(yv.dim)(0.0)
        else VectorD((0 until yv.dim).map(i => (yv(i) - yMin) / range).toIndexedSeq)

    // If your pipeline wants (n x 1) MatrixD:
    val y = new MatrixD(yv.dim, 1)
    for i <- 0 until yv.dim do y(i, 0) = yScaledV(i)

    // run model 
    Optimizer.hp("eta")   = 0.001                                  // set the learning rate (large for small dataset)
    Optimizer.hp("bSize") = 6.0                                  // set the batch size (small for small dataset)
//  val mod = new NeuralNet_XL (x, y)                            // create NeuralNet_XL model with sigmoid (default)
    val mod = new NeuralNet_2L (x, y, f = ActivationFun.f_tanh)   // create NeuralNet_XL model with tanh-tanh-id

    banner ("Small Example - NeuralNet_XL: trainNtest")
    mod.trainNtest ()()                                          // train and test the model
    mod.opti.plotLoss ("NeuralNet_XL")                           // loss function vs epochs

    banner ("Small Example - NeuralNet_XL: trainNtest2")
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
    mod.opti.plotLoss ("NeuralNet_XL")                           // loss function vs epochs
    println (mod.summary2 ())                                    // parameter/coefficient statistics

    banner ("Validation")
    mod.validate ()()

    // banner ("neuralNet_XLTest: Compare with Linear Regression - first column of y")
    // val rg0 = new Regression (x, y0)                             // create a Regression model
    // rg0.trainNtest ()()                                          // train and test the model
    // println (rg0.summary ())                                     // parameter/coefficient statistics

    // banner ("neuralNet_XLTest: Compare with Linear Regression - second column of y")
    // val rg1 = new Regression (x, y1)                             // create a Regression model
    // rg1.trainNtest ()()                                          // train and test the model
    // println (rg1.summary ())                                     // parameter/coefficient statistics