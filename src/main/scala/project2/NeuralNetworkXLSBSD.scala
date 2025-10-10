

package project2

import scala.io.Source
import scalation.mathstat.{VectorD, MatrixD}
// import scalation.modeling.qk

// import scala.math.max
// import scala.runtime.ScalaRunTime.stringOf

// import scalation.mathstat._

import scalation.modeling.ActivationFun
import scalation.modeling.neuralnet.NeuralNet_XL
// import scalation.modeling.Initializer
import scalation.modeling.neuralnet.Optimizer

@main def NeuralNetworkXLSBSD(): Unit =
    // val ox_fname = Array ("mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin")

    
    val filePath = "/mnt/c/Libs/scalation_2.0/data/SeoulBikeData.csv"

    // Load CSV and skip header
    val data: Array[Array[String]] = Source.fromFile(filePath)
        .getLines()
        .drop(1)
        .map(_.split(","))
        .filter(row => row.forall(_.nonEmpty))
        .toArray

    // ================================================================
    // Dependent variable  Y  =  "Rented Bike Count"  (column 1)
    // ================================================================
    val yv = VectorD(data.map(_(1).toDouble))   // column index 1
    val y  = new MatrixD(yv.dim, 1)
    for (i <- 0 until yv.dim) y(i, 0) = yv(i)

    // ================================================================
    // Independent variables  X  =  all columns except Date (0) and Rented Bike Count (1)
    // ================================================================

    // Drop Date (col 0) and Rented Bike Count (col 1)
    val rawX = data.map(row => row.slice(2, row.length))   // keep from Hour onward

    // After dropping first two columns:
    // Hour(0), Temperature(1), Humidity(2), WindSpeed(3), Visibility(4),
    // DewPoint(5), SolarRadiation(6), Rainfall(7), Snowfall(8),
    // Seasons(9), Holiday(10), FunctioningDay(11)

    // We'll treat 'Seasons' and 'Holiday' as categorical variables.
    val seasonIndex  = 9   // after slicing
    val holidayIndex = 10  // after slicing

    val seasonValues  = rawX.map(_(seasonIndex)).distinct
    val holidayValues = rawX.map(_(holidayIndex)).distinct

    // One-hot encode helper
    def oneHotEncode(value: String, categories: Array[String]): Array[Double] =
    categories.map { cat => 
        if value == cat then 1.0 else 0.0
    }

    // Build numeric + one-hot rows
    val xRows = rawX.map { row =>
        val numeric = row.zipWithIndex.collect {
            case (v, i) if i != seasonIndex && i != holidayIndex && i != 11 =>
                v.toDouble
        }
        val seasonVec  = oneHotEncode(row(seasonIndex), seasonValues)
        val holidayVec = oneHotEncode(row(holidayIndex), holidayValues)
        numeric ++ seasonVec ++ holidayVec
    }

    // Convert to MatrixD
    val x = MatrixD(xRows.map(VectorD(_)).toIndexedSeq)

    // ================================================================
    // Print summary to verify
    // ================================================================
    // println(s"x: ${x.dim} features × ${x.dim2} columns")
    // println(s"y: ${y.dim} × ${y.dim2}")
    // println("First row of x: " + x(0))
    // println("First value of y: " + y(0, 0))

    // run model 
    Optimizer.hp("eta")   = 0.001                                  // set the learning rate (large for small dataset)
    Optimizer.hp("bSize") = 6.0                                  // set the batch size (small for small dataset)
//  val mod = new NeuralNet_XL (x, y)                            // create NeuralNet_XL model with sigmoid (default)
    val mod = NeuralNet_XL.rescale (x, y, f = Array (ActivationFun.f_tanh, ActivationFun.f_tanh, ActivationFun.f_id))   // create NeuralNet_XL model with tanh-tanh-id

    

    banner ("Small Example - NeuralNet_XL: trainNtest2")
    mod.trainNtest ()()                                         // train and test the model - with auto-tuning
    mod.opti.plotLoss ("NeuralNet_XL")                           // loss function vs epochs
    println (mod.summary2 ())                                    // parameter/coefficient statistics

    banner ("AutoMPG - NeuralNet_XL: TnT validate")
    mod.validate ()()

    // banner ("neuralNet_XLTest: Compare with Linear Regression - first column of y")
    // val rg0 = new Regression (x, y0)                             // create a Regression model
    // rg0.trainNtest ()()                                          // train and test the model
    // println (rg0.summary ())                                     // parameter/coefficient statistics

    // banner ("neuralNet_XLTest: Compare with Linear Regression - second column of y")
    // val rg1 = new Regression (x, y1)                             // create a Regression model
    // rg1.trainNtest ()()                                          // train and test the model
    // println (rg1.summary ())                                     // parameter/coefficient statistics