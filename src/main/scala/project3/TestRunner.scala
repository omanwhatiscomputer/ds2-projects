package project3

import scalation.{time, banner}
import scalation.mathstat.{VectorD, MatrixD, DeviceConfig}

@main def TestRunner(): Unit =

  val N       = 10_000_000  // vector-vector/scalar ops  (was 1M, GPU slower below ~10M)
  val ROWS    = 3_000       // matrix element-wise/scalar/vec ops  (was 1K)
  val COLS    = 3_000
  val MM      = 1_000       // GEMM * and GEMV  (was 500, first-call CUDA JIT + small data caused GPU to look slower)
  val MM_MUL  = 500         // GEMM mul — naive CPU fallback is O(n^3) ~2.6s at 500; keeping small to avoid ~20s CPU run at 1K

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

  // =========================================================================
  // Vector-Vector Operations  (N = 10,000,000)
  // =========================================================================

  banner("Vector Add (N = 10,000,000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { x + y }
  DeviceConfig.useGPU = false; println("[CPU]"); time { x + y }

  banner("Vector Sub (N = 10,000,000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { x - y }
  DeviceConfig.useGPU = false; println("[CPU]"); time { x - y }

  banner("Vector Mul (N = 10,000,000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { x * y }
  DeviceConfig.useGPU = false; println("[CPU]"); time { x * y }

  banner("Vector Div (N = 10,000,000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { x / y }
  DeviceConfig.useGPU = false; println("[CPU]"); time { x / y }

  // =========================================================================
  // Vector-Scalar Operations  (N = 10,000,000)
  // =========================================================================

  banner("Vector Add Scalar (N = 10,000,000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { x + 10.0 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { x + 10.0 }

  banner("Vector Sub Scalar (N = 10,000,000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { x - 10.0 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { x - 10.0 }

  banner("Vector Mul Scalar (N = 10,000,000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { x * 2.0 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { x * 2.0 }

  banner("Vector Div Scalar (N = 10,000,000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { x / 2.0 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { x / 2.0 }

  // =========================================================================
  // Matrix Element-wise Operations  (3,000 x 3,000)
  // =========================================================================

  banner("Matrix Add (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 + m2 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 + m2 }

  banner("Matrix Sub (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 - m2 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 - m2 }

  banner("Matrix Element-wise Mul (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 *~ m2 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 *~ m2 }

  banner("Matrix Div (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 / m2 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 / m2 }

  // =========================================================================
  // Matrix-Scalar Operations  (3,000 x 3,000)
  // =========================================================================

  banner("Matrix Add Scalar (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 + 10.0 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 + 10.0 }

  banner("Matrix Sub Scalar (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 - 1.0 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 - 1.0 }

  banner("Matrix Mul Scalar (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 * 2.0 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 * 2.0 }

  banner("Matrix Div Scalar (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 / 2.0 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 / 2.0 }

  // =========================================================================
  // Matrix op Row Vector  (3,000 x 3,000)
  // =========================================================================

  banner("Matrix Add Row Vector (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 + rowVec }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 + rowVec }

  banner("Matrix Sub Row Vector (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 - rowVec }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 - rowVec }

  banner("Matrix Mul Row Vector (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 *~ rowVec }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 *~ rowVec }

  banner("Matrix Div Row Vector (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 / rowVec }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 / rowVec }

  // =========================================================================
  // Matrix op Col Vector  (3,000 x 3,000)
  // =========================================================================

  banner("Matrix Add Col Vector (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 +^ colVec }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 +^ colVec }

  banner("Matrix Sub Col Vector (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1 -^ colVec }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1 -^ colVec }

  banner("Matrix Mul Col Vector (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { colVec *~: m1 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { colVec *~: m1 }

  // =========================================================================
  // Matrix Multiplication — GEMM
  // =========================================================================

  // mul uses the naive CPU fallback (O(n^3) without tiling) — keep at MM_MUL=500
  // to avoid ~20s CPU run; already shows clear GPU advantage
  banner("GEMM Matrix mul Matrix (500 x 500)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { gms1 mul gms2 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { gms1 mul gms2 }

  // * uses tiled CPU fallback — run at MM=1000; also warms the matrixMulAddr lazy val
  // so the GPU time below is real computation, not first-call CUDA JIT overhead
  banner("GEMM Matrix * Matrix (1000 x 1000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { gm1 * gm2 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { gm1 * gm2 }

  // =========================================================================
  // Matrix-Vector Multiply — GEMV  (1,000 x 1,000)
  // =========================================================================

  banner("GEMV Matrix * Vector (1000 x 1000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { gm1 * gemvVec }
  DeviceConfig.useGPU = false; println("[CPU]"); time { gm1 * gemvVec }

  banner("Transpose GEMV Vector *: Matrix (1000 x 1000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { gemvVec *: gm1 }
  DeviceConfig.useGPU = false; println("[CPU]"); time { gemvVec *: gm1 }

  banner("Transpose GEMV Matrix dot Vector (1000 x 1000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { gm1 dot gemvVec }
  DeviceConfig.useGPU = false; println("[CPU]"); time { gm1 dot gemvVec }

  // =========================================================================
  // Reductions  (3,000 x 3,000)
  // =========================================================================

  banner("Global Sum (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.sum }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1.sum }

  banner("Global Max (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.mmax }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1.mmax }

  banner("Global Min (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.mmin }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1.mmin }

  banner("Column Sum (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.sumV }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1.sumV }

  banner("Column Max (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.max }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1.max }

  banner("Column Min (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.min }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1.min }

  banner("Row Sum (3000 x 3000)")
  DeviceConfig.useGPU = true;  println("[GPU]"); time { m1.sumVr }
  DeviceConfig.useGPU = false; println("[CPU]"); time { m1.sumVr }

end TestRunner
