package project1.SBSD
import scala.io.Source
import scalation.mathstat.{VectorD, MatrixD, PlotM}
import scalation.modeling.{Regression}
import scalation.modeling.qk

@main def SimpleRegression3(): Unit =
    val ox_fname = Array ("Date","Rented Bike Count","Hour","Temperature(C)","Humidity(%)","Wind speed (m/s)","Visibility (10m)","Dew point temperature(C)","Solar Radiation (MJ/m2)","Rainfall(mm)","Snowfall (cm)","Seasons","Holiday","Functioning Day")
    
    val filePath = "/mnt/c/Libs/scalation_2.0/data/SeoulBikeData.csv"

    val data: Array[Array[String]] = Source.fromFile(filePath)
        .getLines()
        .drop(1)                           // skip header
        .map(_.split(","))
        .filter(row => row.forall(_.nonEmpty))
        .toArray

    // Dependent variable (last column: Functioning Day → binary)
    val y = VectorD(
        data.map { row =>
            if row.last.trim == "Yes" then 1.0 else 0.0
        }
    )

    // Independent variables: drop Date (col 0) and Functioning Day (last col)
    val rawX = data.map(row => row.slice(1, row.length - 1))

    // Categorical indices after dropping Date
    val seasonIndex  = 10   // Seasons
    val holidayIndex = 11   // Holiday

    // Extract unique categories for one-hot encoding
    val seasonValues  = rawX.map(_(seasonIndex)).distinct
    val holidayValues = rawX.map(_(holidayIndex)).distinct

    // One-hot encode categorical variables
    def oneHotEncode(value: String, categories: Array[String]): Array[Double] =
        categories.map { cat =>
            if value == cat then 1.0 else 0.0
        }

    val xRows = rawX.map { row =>
        val numeric = row.zipWithIndex.collect {
            case (v, i) if i != seasonIndex && i != holidayIndex => v.toDouble
        }
        val seasonVec  = oneHotEncode(row(seasonIndex), seasonValues)
        val holidayVec = oneHotEncode(row(holidayIndex), holidayValues)
        numeric ++ seasonVec ++ holidayVec
    }

    val x = MatrixD(xRows.map(VectorD(_)).toIndexedSeq)

    val mod = new Regression (x, y, ox_fname)                          // create a simple regression model
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
