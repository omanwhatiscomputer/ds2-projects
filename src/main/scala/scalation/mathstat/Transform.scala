
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Thu Mar 13 14:06:11 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Support for Transformation Functions with their Inverse
 *
 *  https://www.infoq.com/news/2023/10/foreign-function-and-memory-api/
 */

package scalation
package mathstat

import scala.math._

import VectorDOps._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Thee `⚬` extension method performs function composition f ⚬ g.
 *  @see www.scala-lang.org/api/current/scala/Function1.html
 *  @tparam A  the type to which function `g` can be applied
 *  @tparam B  the type to which function `f` can be applied
 *  @tparam R  the return type for f ⚬ g
 */
extension [A, B, R](f: B => R)

    def ⚬ (g: A => B): A => R = f compose g


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Transform` trait supports the use of transformation functions, such that it
 *  is easy to take the inverse transform.  When a transformation uses arguments,
 *  they are remembered for use by the inverse transformation.
 *  @param x  the vector or matrix being transformed
 */
trait Transform (x: VectorD | MatrixD = null):

    protected var lu: VectorD = VectorD (2, 5)                           // optional default range/bounds [l .. u]
    protected var b: MatrixD  = null                                     // optional argument matrix

    if x != null then
        x match
            case xM: MatrixD => setB (xM)
            case xV: VectorD => setB (MatrixD (xV).transpose)

    def setLU (_lu: VectorD): Unit = lu = _lu                            // set the default bounds
    def setB (x: MatrixD): Unit = b = x                                  // set the argument matrix 
    def f  (x: MatrixD): MatrixD                                         // transformation function
    def fi (y: MatrixD): MatrixD                                         // inverse transformation function

    val f:  FunctionV2V = (x: VectorD) => f(MatrixD(x).transpose)(?, 0)
    val fi: FunctionV2V = (y: VectorD) => fi(MatrixD(y).transpose)(?, 0)

    def df (x: VectorD): MatrixD = null                                  // partial derivative of f

    def df (x: MatrixD): MatrixD =                                       // column-by-column partial derivative of f wrt to w
        var jMatrix = df (x(?, 0))
        for j <- 1 until x.dim2 do jMatrix = jMatrix ++^ df (x(?, j))
        jMatrix
    end df

    def df (x: MatrixD, i: Int): MatrixD =                               // partial derivative of each column wrt wi
        if i == 0 || i == 1 then
            var jMatrix = MatrixD (df (x(?, 0))(?, i)).transpose
            for j <- 1 until x.dim2 do jMatrix = jMatrix :^+ df (x(?, j))(?, i)
            jMatrix
        else
            df(x)
    end df

end Transform


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `zForm` class applies the z-transformation (subtract mean and divide by standard deviation).
 */
class zForm (x: VectorD | MatrixD) extends Transform (x):
    override def setB (x: MatrixD): Unit = b = x.mu_sig
    def f (x: MatrixD): MatrixD  = (x - b(0)) / b(1)
    def fi (y: MatrixD): MatrixD = (y *~ b(1)) + b(0)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `rangeForm` class transforms values to the default range/bounds lu.
 */
class rangeForm (x: VectorD | MatrixD) extends Transform (x):
    override def setB (x: MatrixD): Unit = b = x.min_max
    def f (x: MatrixD): MatrixD  = (x - b(0)) * (lu(1) - lu(0)) / (b(1) - b(0))  + lu(0)
    def fi (y: MatrixD): MatrixD = (y - lu(0)) *~ (b(1) - b(0)) /(lu(1) - lu(0)) + b(0)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `logForm` object applies the log-transformation.
 */
object logForm extends Transform ():
    def f (x: MatrixD): MatrixD  = x.log
    def fi (y: MatrixD): MatrixD = y.exp

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `logForm` class applies a scaled and shifted log-transformation.
 */
class logForm (x: VectorD = VectorD (1.0, 0.25)) extends Transform (x):
    def f (x: MatrixD): MatrixD  = x.map_ (z => log (b(0, 0) * z + b(1, 0)))
    def fi (y: MatrixD): MatrixD = y.map_ (z => (exp (z)  - b(1, 0)) / b(0, 0))
    override def df (x: VectorD): MatrixD = MatrixD (x / (b(0, 0) * x + b(1, 0)), 1 / (b(0, 0) * x + b(1, 0))).transpose

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `log1pForm` object applies the log1p-transformation (log (z+1)).
 */
object log1pForm extends Transform ():
    def f (x: MatrixD): MatrixD  = x.log1p
    def fi (y: MatrixD): MatrixD = y.expm1

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cosForm` class applies the cosine-transformation.
 */
class cosForm (x: VectorD = VectorD (0.0, 0.25)) extends Transform (x):
    def f (x: MatrixD): MatrixD  = x.map_ (z => cos ((b(0, 0) * z + b(1, 0)) * Piby2))
    def fi (y: MatrixD): MatrixD = y.map_ (z => (acos (z) * _2byPi - b(1, 0)) / b(0, 0))
    override def df (x: VectorD): MatrixD = MatrixD (x.map (z => -sin ((b(0, 0) * z + b(1, 0)) * Piby2)) * (Piby2 * x),
                                                     x.map (z => -sin ((b(0, 0) * z + b(1, 0)) * Piby2)) * Piby2).transpose

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `sinForm` class applies the sine-transformation.
 */
class sinForm (x: VectorD = VectorD (0.0, 0.25)) extends Transform (x):
    def f (x: MatrixD): MatrixD  = x.map_ (z => sin ((b(0, 0) * z + b(1, 0)) * Piby2))
    def fi (y: MatrixD): MatrixD = y.map_ (z => (asin (z) * _2byPi - b(1, 0)) / b(0, 0))
    override def df (x: VectorD): MatrixD = MatrixD (x.map (z => cos ((b(0, 0) * z + b(1, 0)) * Piby2) * (Piby2 * z)) ,
                                                     x.map (z => cos ((b(0, 0) * z + b(1, 0)) * Piby2) * Piby2 )).transpose

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `powForm` class applies the power-transformation x^p for power p > 1.
 */
class powForm (x: VectorD = VectorD (0.0, 2.0)) extends Transform (x):
    def f (x: MatrixD): MatrixD  = (x + b(0, 0) ) ~^ b(1, 0)
    def fi (y: MatrixD): MatrixD = y ~^ (1/b(1, 0)) - b(0, 0)
    override def df (x: VectorD): MatrixD = MatrixD (((x + b(0, 0)) ~^ (b(1, 0) - 1)) * b(1, 0),
                                                     f(x) * (x + b(0, 0)).map (z => log(z))).transpose


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `transformTest` tests the `Transform` class at the vector level.
 *  > runMain scalation.mathstat.transformTest
 */
@main def transformTest (): Unit =

    val x = VectorD (1, 2, 3)
    println (s"x = $x")

    banner ("zForm Transformation")
    val zForm1 = zForm (x)                                                    // set the argument vector
    var y = zForm1.f (x)
    var z = zForm1.fi (y)
    println (s"y = $y, z = $z")
    println (s"df: ${zForm1.df (x)}")

    banner ("rangeForm Transformation")
    val rangeForm1 = rangeForm (x)                                            // set the argument vector
    y = rangeForm1.f (x)
    z = rangeForm1.fi (y)
    println (s"y = $y, z = $z")

    banner ("logForm Transformation")
    y = logForm.f (x)
    z = logForm.fi (y)
    println (s"y = $y, z = $z")

    banner ("log1pForm Transformation")
    y = log1pForm.f (x)
    z = log1pForm.fi (y)
    println (s"y = $y, z = $z")

    banner("logForm1 Transformation")
    val logForm1 = logForm (VectorD (1.0, 2.0))                               // set the argument vector
    y = logForm1.f (x)
    z = logForm1.fi (y)
    println (s"y = $y, z = $z")
    println (s"df: ${logForm1.df (x)}")

    banner ("cosForm Transformation")
    val cosForm1 = cosForm (VectorD (1.0, 0.25))                              // set the argument vector
    y = cosForm1.f (x)
    z = cosForm1.fi (y)
    println (s"y = $y, z = $z")
    println (s"df: ${cosForm1.df (x)}")

    banner ("sinForm Transformation")
    val sinForm1 = sinForm (VectorD (1.0, 0.25))                              // set the argument vector
    y = sinForm1.f (x)
    z = sinForm1.fi (y)
    println (s"y = $y, z = $z")
    println (s"df: ${sinForm1.df (x)}")

    banner ("powForm Transformation")
    val powForm1 = powForm (VectorD (1.0, 2.0))                               // set the argument vector
    y = powForm1.f (x)
    z = powForm1.fi (y)
    println (s"y = $y, z = $z")
    println (s"df: ${powForm1.df (x)}")

end transformTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `transformTest2` tests the `Transform` class at the matrix level.
 *  > runMain scalation.mathstat.transformTest2
 */
@main def transformTest2 (): Unit =

    val x = MatrixD ((3, 2), 3, 1,
                             5, 2,
                             6, 3)
    println (s"x = $x")

    banner ("zForm Transformation")
    val zForm1 = zForm (x)                                                    // set the argument vector
    var y = zForm1.f (x)
    var z = zForm1.fi (y)
    println (s"y = $y, z = $z")

    banner ("rangeForm Transformation")
    val rangeForm1 = rangeForm (x)                                            // set the argument vector
    y = rangeForm1.f (x)
    z = rangeForm1.fi (y)
    println (s"y = $y, z = $z")

    banner ("logForm Transformation")
    y = logForm.f (x)
    z = logForm.fi (y)
    println (s"y = $y, z = $z")

    banner ("log1pForm Transformation")
    y = log1pForm.f (x)
    z = log1pForm.fi (y)
    println (s"y = $y, z = $z")

    banner("logForm1 Transformation")
    val logForm1 = logForm(VectorD (1.0, 2.0))                                // set the argument vector
    y = logForm1.f(x)
    z = logForm1.fi(y)
    println (s"y = $y, z = $z")
    println (s"df: ${logForm1.df (x)}")

    banner ("cosForm Transformation")
    val cosForm1 = cosForm (VectorD (1.0, 0.25))                              // set the argument vector
    y = cosForm1.f (x)
    z = cosForm1.fi (y)
    println (s"y = $y, z = $z")
    println (s"df: ${cosForm1.df (x)}")

    banner ("sinForm Transformation")
    val sinForm1 = sinForm (VectorD (1.0, 0.25))                              // set the argument vector
    y = sinForm1.f (x)
    z = sinForm1.fi (y)
    println (s"y = $y, z = $z")
    println (s"df: ${sinForm1.df(x)}")

    banner ("powForm Transformation")
    val powForm1 = powForm (VectorD (1.0, 2.0))                               // set the argument vector
    y = powForm1.f (x)
    z = powForm1.fi (y)
    println (s"y = $y, z = $z")
    println (s"df: ${powForm1.df (x)}")

end transformTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `transformTest3` tests the `Transform` class at the matrix level.
 *  > runMain scalation.mathstat.transformTest3
 */
@main def transformTest3 (): Unit =

    val x = VectorD (3, 5, 6, 2, 1, 3, 2, 4, 6, 87, 1000)
    println (s"x = $x")

    banner ("zForm Transformation")
    val zForm1 = zForm (x)                                                    // set the argument vector
    var y = zForm1.f (x)
    var z = zForm1.fi (y)
    println (s"y = $y, \nz = $z")

    banner ("powForm Transformation")
    val powForm1 = powForm (VectorD (0.0, 1.5))                               // set the argument vector
    y = powForm1.f (x)
    z = powForm1.fi (y)
    println (s"y = $y, \nz = $z")

    val fsc = (zForm1.f(_: VectorD)) ⚬ (powForm1.f(_: VectorD)) ⚬ (zForm1.fi(_: VectorD))
    val ysc = fsc (y)
    println (s"ysc = ${ysc}")

    val ysc2 = fsc (y(0 until 3))
    println (s"ysc = ${ysc2}")

end transformTest3

