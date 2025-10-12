

package project2

import scalation.mathstat.{MatrixD, VectorD}

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
// import scalation.modeling.qk

// import scala.math.max
// import scala.runtime.ScalaRunTime.stringOf

// import scalation.mathstat._

import scalation.modeling.ActivationFun
import scalation.modeling.neuralnet.NeuralNet_3L
// import scalation.modeling.Initializer
import scalation.modeling.neuralnet.Optimizer

@main def NeuralNetwork3LSBSD(): Unit =
    // val ox_fname = Array ("mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin")

    
    val filePath = "C:/Libs/scalation_2.0/data/SeoulBikeData.csv"

    // Load CSV and skip header
    val data: Array[Array[String]] =
      Source.fromFile(filePath)
        .getLines()
        .drop(1) // drop header
        .map(_.trim.split(","))
        .filter(row => row.forall(_.nonEmpty))
        .toArray

    // -----------------------------
    // y: "Rented Bike Count" (col 1)
    // -----------------------------
    val yv = VectorD(data.map(_(1).toDouble))
    val y = new MatrixD(yv.dim, 1)
    for (i <- 0 until yv.dim) y(i, 0) = yv(i)

    // -----------------------------------------------------------
    // X: drop Date(0) + Rented Bike Count(1) => slice from index 2
    // After slicing, columns are reindexed 0..11 as:
    //  0 Hour, 1 Temperature(C), 2 Humidity(%), 3 Wind speed (m/s),
    //  4 Visibility (10m), 5 Dew point temperature(C),
    //  6 Solar Radiation (MJ/m2), 7 Rainfall(mm), 8 Snowfall (cm),
    //  9 Seasons, 10 Holiday, 11 Functioning Day
    // -----------------------------------------------------------
    val rawX = data.map(row => row.slice(2, row.length))

    // After dropping first two columns:
    // Hour(0), Temperature(1), Humidity(2), WindSpeed(3), Visibility(4),
    // DewPoint(5), SolarRadiation(6), Rainfall(7), Snowfall(8),
    // Seasons(9), Holiday(10), FunctioningDay(11)

    val seasonIdx = 9
    val holidayIdx = 10
    val functIdx = 11

    // Stable category lists (sorted for determinism)
    val seasonCats = rawX.map(_(seasonIdx)).distinct.sorted
    val holidayCats = rawX.map(_(holidayIdx)).distinct.sorted
    // Functioning Day is binary "Yes"/"No" – map directly to 1/0
    // (If you prefer one-hot for Functioning Day, do the same pattern as below.)

    def oneHot(value: String, cats: Array[String]): Array[Double] =
      cats.map(cat => if (value == cat) then 1.0 else 0.0)

    def toDoubleSafe(s: String): Double =
      s.trim.toDouble

    val xRows: Array[Array[Double]] = rawX.map { row =>
      // numeric features: everything except categorical indices
      val numeric = ArrayBuffer[Double]()
      var i = 0
      while (i < row.length) do {
        if (i != seasonIdx && i != holidayIdx && i != functIdx) then {
          numeric += toDoubleSafe(row(i))
        }
        i += 1
      }

      // one-hots
      val seasonOH = oneHot(row(seasonIdx), seasonCats)
      val holidayOH = oneHot(row(holidayIdx), holidayCats)

      // binary for Functioning Day: Yes -> 1.0, No -> 0.0 (case-insensitive)
      val funcVal = row(functIdx).trim.toLowerCase match {
        case "yes" => 1.0
        case "no" => 0.0
        case other => // fallback if dataset has variants
          if (other.startsWith("y")) then 1.0 else 0.0
      }

      // concatenate safely (avoid Array ++ type widening)
      Array.concat(numeric.toArray, seasonOH, holidayOH, Array(funcVal))
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
    val mod = NeuralNet_3L.rescale (x, y, f = ActivationFun.f_tanh)   // create NeuralNet_XL model with tanh-tanh-id

    

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