package scalation
package mathstat

import scala.util.Try

import uk.ac.manchester.tornado.api.enums.DataTransferMode
import uk.ac.manchester.tornado.api.annotations.Parallel
import uk.ac.manchester.tornado.api.common.TornadoFunctions
import uk.ac.manchester.tornado.api.{TaskGraph, TornadoExecutionPlan}

// ─── TornadoVM kernel methods ─────────────────────────────────────────────────
// All matrices are passed as flat row-major arrays; dimensions are inferred from
// the array lengths so no Int parameters are needed (avoids TornadoVM boxing issues).
object TMKernels:

  // Square GEMM: c = a * b  (n×n each).  n = sqrt(a.length).
  def matMul(a: Array[Double], b: Array[Double], c: Array[Double]): Unit =
    val n = Math.sqrt(a.length.toDouble).toInt
    @Parallel var i = 0
    while i < n do
      @Parallel var j = 0
      while j < n do
        var s = 0.0; var k = 0
        while k < n do { s += a(i * n + k) * b(k * n + j); k += 1 }
        c(i * n + j) = s
        j += 1
      i += 1

  // GEMV: result(rows) = a(rows×cols) * v(cols).  cols = v.length, rows = a.length/cols.
  def matVecMul(a: Array[Double], v: Array[Double], result: Array[Double]): Unit =
    val cols = v.length
    val rows = a.length / cols
    @Parallel var i = 0
    while i < rows do
      var s = 0.0; var j = 0
      while j < cols do { s += a(i * cols + j) * v(j); j += 1 }
      result(i) = s
      i += 1

  // Column sum: result.length = cols, rows = a.length / cols.
  def colSum(a: Array[Double], result: Array[Double]): Unit =
    val cols = result.length
    val rows = a.length / cols
    @Parallel var j = 0
    while j < cols do
      var s = 0.0; var i = 0
      while i < rows do { s += a(i * cols + j); i += 1 }
      result(j) = s
      j += 1

  def colMin(a: Array[Double], result: Array[Double]): Unit =
    val cols = result.length
    val rows = a.length / cols
    @Parallel var j = 0
    while j < cols do
      var mn = Double.MaxValue; var i = 0
      while i < rows do { mn = Math.min(mn, a(i * cols + j)); i += 1 }
      result(j) = mn
      j += 1

  def colMax(a: Array[Double], result: Array[Double]): Unit =
    val cols = result.length
    val rows = a.length / cols
    @Parallel var j = 0
    while j < cols do
      var mx = -Double.MaxValue; var i = 0
      while i < rows do { mx = Math.max(mx, a(i * cols + j)); i += 1 }
      result(j) = mx
      j += 1

// ─── Public dispatch API ──────────────────────────────────────────────────────
object TornadoMatrixOps:

  def isAvailable: Boolean = DeviceConfig.useGPU

  private def setDevice(graphName: String): Unit =
    DeviceConfig.applyDeviceProperty(graphName, "t0")

  /** Square GEMM. Returns flat row-major result; caller wraps into MatrixD. */
  def gemm(a: MatrixD, b: MatrixD): Option[Array[Double]] =
    if !isAvailable || a.dim != a.dim2 || b.dim != b.dim2 || a.dim != b.dim then return None
    val fa = flatten(a); val fb = flatten(b)
    val fc = new Array[Double](a.dim * b.dim2)
    Try {
      setDevice("tmo_gemm")
      val kMul: TornadoFunctions.Task3[Array[Double], Array[Double], Array[Double]] = TMKernels.matMul
      val plan = new TornadoExecutionPlan(
        new TaskGraph("tmo_gemm")
          .transferToDevice(DataTransferMode.FIRST_EXECUTION, fa, fb)
          .task("t0", kMul, fa, fb, fc)
          .transferToHost(DataTransferMode.EVERY_EXECUTION, fc)
          .snapshot())
      plan.execute()
      fc
    }.toOption

  /** GEMV. Returns flat result vector; caller wraps into VectorD. */
  def gemv(a: MatrixD, y: VectorD): Option[Array[Double]] =
    if !isAvailable then return None
    val fa  = flatten(a)
    val fv  = y.v
    val res = new Array[Double](a.dim)
    Try {
      setDevice("tmo_gemv")
      val kGemv: TornadoFunctions.Task3[Array[Double], Array[Double], Array[Double]] = TMKernels.matVecMul
      val plan = new TornadoExecutionPlan(
        new TaskGraph("tmo_gemv")
          .transferToDevice(DataTransferMode.FIRST_EXECUTION, fa, fv)
          .task("t0", kGemv, fa, fv, res)
          .transferToHost(DataTransferMode.EVERY_EXECUTION, res)
          .snapshot())
      plan.execute()
      res
    }.toOption

  /** Column sum. Returns flat result; caller wraps into VectorD. */
  def colSum(a: MatrixD): Option[Array[Double]] = colOp("tmo_csum", TMKernels.colSum, a)

  /** Column min. Returns flat result; caller wraps into VectorD. */
  def colMin(a: MatrixD): Option[Array[Double]] = colOp("tmo_cmin", TMKernels.colMin, a)

  /** Column max. Returns flat result; caller wraps into VectorD. */
  def colMax(a: MatrixD): Option[Array[Double]] = colOp("tmo_cmax", TMKernels.colMax, a)

  // ── Utilities ─────────────────────────────────────────────────────────────

  /** Flatten jagged MatrixD backing array into a row-major flat array. */
  def flatten(m: MatrixD): Array[Double] =
    val flat = new Array[Double](m.dim * m.dim2)
    var i = 0
    while i < m.dim do
      System.arraycopy(m.v(i), 0, flat, i * m.dim2, m.dim2)
      i += 1
    flat

  /** Reconstruct jagged Array[Array[Double]] from flat row-major array. */
  def unflatten(flat: Array[Double], rows: Int, cols: Int): Array[Array[Double]] =
    val a = Array.ofDim[Double](rows, cols)
    var i = 0
    while i < rows do
      System.arraycopy(flat, i * cols, a(i), 0, cols)
      i += 1
    a

  private type ColKernel = TornadoFunctions.Task2[Array[Double], Array[Double]]

  private def colOp(name: String, k: ColKernel, a: MatrixD): Option[Array[Double]] =
    if !isAvailable then return None
    val fa  = flatten(a)
    val res = new Array[Double](a.dim2)
    Try {
      setDevice(name)
      val plan = new TornadoExecutionPlan(
        new TaskGraph(name)
          .transferToDevice(DataTransferMode.FIRST_EXECUTION, fa)
          .task("t0", k, fa, res)
          .transferToHost(DataTransferMode.EVERY_EXECUTION, res)
          .snapshot())
      plan.execute()
      res
    }.toOption
