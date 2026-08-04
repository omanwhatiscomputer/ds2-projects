package project3

import scalation.{time, banner}
import scalation.mathstat.{VectorD, MatrixD, TensorD, DeviceConfig}

@main def TestRunner(): Unit =

  val N          = 10_000_000
  val ROWS       = 3_000
  val COLS       = 3_000
  val MM         = 1_000
  val MM_MUL     = 500
  val TD         = 100
  val TC         = 20
  val iterations = 20

  def fmt(n: Int): String = "%,d".format(n)

  // ---- Build test data ----
  val x       = VectorD(Array.fill(N)(1.5))
  val y       = VectorD(Array.fill(N)(2.5))
  val rowVec  = VectorD(Array.fill(COLS)(1.5))
  val colVec  = VectorD(Array.fill(ROWS)(2.5))
  val gemvVec = VectorD(Array.fill(MM)(1.5))

  val m1    = MatrixD((0 until ROWS).map(_ => VectorD(Array.fill(COLS)(1.5))))
  val m2    = MatrixD((0 until ROWS).map(_ => VectorD(Array.fill(COLS)(2.5))))
  val gm1   = MatrixD((0 until MM).map(_ => VectorD(Array.fill(MM)(1.5))))
  val gm2   = MatrixD((0 until MM).map(_ => VectorD(Array.fill(MM)(2.5))))
  val gms1  = MatrixD((0 until MM_MUL).map(_ => VectorD(Array.fill(MM_MUL)(1.5))))
  val gms2  = MatrixD((0 until MM_MUL).map(_ => VectorD(Array.fill(MM_MUL)(2.5))))

  val t1  = TensorD.fill(TD, TD, TD, 1.5)
  val t2  = TensorD.fill(TD, TD, TD, 2.5)
  val tb  = MatrixD((0 until TC).map(_ => VectorD(Array.fill(TC)(1.5))))
  val tc  = MatrixD((0 until TC).map(_ => VectorD(Array.fill(TC)(1.5))))
  val td  = MatrixD((0 until TC).map(_ => VectorD(Array.fill(TC)(1.5))))
  val ts  = TensorD.fill(TC, TC, TC, 1.5)

  // =========================================================================
  // Vector-Vector Operations
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"Vector Add (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x + y }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x + y }

      banner(s"Vector Sub (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x - y }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x - y }

      banner(s"Vector Mul (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x * y }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x * y }

      banner(s"Vector Div (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x / y }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x / y }

  // =========================================================================
  // Vector-Scalar Operations
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"Vector Add Scalar (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x + 10.0 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x + 10.0 }

      banner(s"Vector Sub Scalar (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x - 10.0 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x - 10.0 }

      banner(s"Vector Mul Scalar (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x * 2.0 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x * 2.0 }

      banner(s"Vector Div Scalar (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x / 2.0 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x / 2.0 }

  // =========================================================================
  // Matrix Element-wise Operations
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"Matrix Add ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 + m2 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 + m2 }

      banner(s"Matrix Sub ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 - m2 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 - m2 }

      banner(s"Matrix Element-wise Mul ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 *~ m2 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 *~ m2 }

      banner(s"Matrix Div ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 / m2 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 / m2 }

  // =========================================================================
  // Matrix-Scalar Operations
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"Matrix Add Scalar ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 + 10.0 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 + 10.0 }

      banner(s"Matrix Sub Scalar ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 - 1.0 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 - 1.0 }

      banner(s"Matrix Mul Scalar ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 * 2.0 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 * 2.0 }

      banner(s"Matrix Div Scalar ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 / 2.0 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 / 2.0 }

  // =========================================================================
  // Matrix op Row Vector
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"Matrix Add Row Vector ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 + rowVec }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 + rowVec }

      banner(s"Matrix Sub Row Vector ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 - rowVec }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 - rowVec }

      banner(s"Matrix Mul Row Vector ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 *~ rowVec }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 *~ rowVec }

      banner(s"Matrix Div Row Vector ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 / rowVec }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 / rowVec }

  // =========================================================================
  // Matrix op Col Vector
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"Matrix Add Col Vector ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 +^ colVec }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 +^ colVec }

      banner(s"Matrix Sub Col Vector ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 -^ colVec }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1 -^ colVec }

      banner(s"Matrix Mul Col Vector ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { colVec *~: m1 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { colVec *~: m1 }

  // =========================================================================
  // Matrix Multiplication — GEMM
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      // mul uses the naive CPU fallback (O(n^3) without tiling) — keep at MM_MUL to avoid long CPU run
      banner(s"GEMM Matrix mul Matrix ($MM_MUL x $MM_MUL)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { gms1 mul gms2 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { gms1 mul gms2 }

      // * uses tiled CPU fallback — also warms the matrixMulAddr lazy val
      banner(s"GEMM Matrix * Matrix ($MM x $MM)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { gm1 * gm2 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { gm1 * gm2 }

  // =========================================================================
  // Matrix-Vector Multiply — GEMV
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"GEMV Matrix * Vector ($MM x $MM)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { gm1 * gemvVec }
      DeviceConfig.useGPU = false; println("[CPU]"); time { gm1 * gemvVec }

      banner(s"Transpose GEMV Vector *: Matrix ($MM x $MM)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { gemvVec *: gm1 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { gemvVec *: gm1 }

      banner(s"Transpose GEMV Matrix dot Vector ($MM x $MM)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { gm1 dot gemvVec }
      DeviceConfig.useGPU = false; println("[CPU]"); time { gm1 dot gemvVec }

  // =========================================================================
  // Reductions
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"Global Sum ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.sum }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1.sum }

      banner(s"Global Max ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.mmax }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1.mmax }

      banner(s"Global Min ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.mmin }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1.mmin }

      banner(s"Column Sum ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.sumV }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1.sumV }

      banner(s"Column Max ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.max }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1.max }

      banner(s"Column Min ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.min }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1.min }

      banner(s"Row Sum ($ROWS x $COLS)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.sumVr }
      DeviceConfig.useGPU = false; println("[CPU]"); time { m1.sumVr }

  // =========================================================================
  // VectorD Reductions
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"Vector Sum (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x.sum }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x.sum }

      banner(s"Vector Min (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x.min }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x.min }

      banner(s"Vector Max (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x.max }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x.max }

      banner(s"Vector Dot (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x dot y }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x dot y }

      banner(s"Vector NormSq (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x.normSq }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x.normSq }

      banner(s"Vector Norm (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x.norm }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x.norm }

      banner(s"Vector Norm1 (N = ${fmt(N)})")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { x.norm1 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { x.norm1 }

  // =========================================================================
  // TensorD Operations
  // =========================================================================

  for i <- 0 until iterations do
    if i == iterations - 1 then
      banner(s"Tensor Add ($TD x $TD x $TD)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { t1 + t2 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { t1 + t2 }

      banner(s"Tensor Sub ($TD x $TD x $TD)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { t1 - t2 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { t1 - t2 }

      banner(s"Tensor Mul ($TD x $TD x $TD)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { t1 * t2 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { t1 * t2 }

      banner(s"Tensor Div ($TD x $TD x $TD)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { t1 / t2 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { t1 / t2 }

      banner(s"Tensor Add Scalar ($TD x $TD x $TD)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { t1 + 10.0 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { t1 + 10.0 }

      banner(s"Tensor Mul Scalar ($TD x $TD x $TD)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { t1 * 2.0 }
      DeviceConfig.useGPU = false; println("[CPU]"); time { t1 * 2.0 }

      banner(s"Tensor Global Sum ($TD x $TD x $TD)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { t1.sum }
      DeviceConfig.useGPU = false; println("[CPU]"); time { t1.sum }

      banner(s"Tensor Contraction ($TC x $TC x $TC)")
      DeviceConfig.useGPU = true;  println("[GPU]"); time { ts * (tb, tc, td) }
      DeviceConfig.useGPU = false; println("[CPU]"); time { ts * (tb, tc, td) }

end TestRunner
