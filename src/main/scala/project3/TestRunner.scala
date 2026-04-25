package project3

import scalation.mathstat.{DeviceConfig, MatrixD, VectorD}

/** Benchmark all TornadoVM-accelerated VectorD and MatrixD operators.
 *  Each section runs the operation twice: once with DeviceConfig.useGPU=true
 *  (TornadoVM dispatches to GPU) and once with useGPU=false (pure CPU path).
 *
 *  Toggle globally with JVM flag:  -Dscalation.useGPU=false
 */
@main def TestRunner(): Unit =

  // ─── Device selection ────────────────────────────────────────────────────
  // TornadoVM prints all available devices at startup (build.sbt sets
  // -Dtornado.device.desc=true).  To target a specific device, set:
  //   DeviceConfig.driverIndex = 0   // 0=OpenCL, 1=PTX
  //   DeviceConfig.deviceIndex = 1   // e.g. 1 = CPU OpenCL device
  // or via JVM flags:  -Dscalation.driver=0 -Dscalation.device=1


  def banner(s: String): Unit = println(s"\n${"═" * 68}\n  $s\n${"═" * 68}")

  def timed[R](label: String)(block: => R): R =
    val t0 = System.nanoTime()
    val r  = block
    println(s"  [$label] ${(System.nanoTime() - t0) / 1_000_000}ms")
    r

  // ─── Sizes ───────────────────────────────────────────────────────────────
  val N    = 10_000_000   // vector length
  val DIM  = 1_000        // matrix size for GEMV + column ops
  val GDIM = 500          // matrix size for GEMM (O(n³) — keep tractable)

  // ─── Test data ───────────────────────────────────────────────────────────
  val v1 = VectorD(Array.tabulate(N)(i => (i % 1_000 + 1).toDouble))
  val v2 = VectorD(Array.fill(N)(2.0))

  val m1  = new MatrixD(DIM, DIM, Array.tabulate(DIM)(i => Array.tabulate(DIM)(j => (i * DIM + j) % 100 + 1.0)))
  val v_m = VectorD(Array.tabulate(DIM)(i => (i % 50 + 0.5)))   // for GEMV

  val g1  = new MatrixD(GDIM, GDIM, Array.tabulate(GDIM)(i => Array.tabulate(GDIM)(j => (i * GDIM + j) % 10 + 1.0)))
  val g2  = new MatrixD(GDIM, GDIM, Array.tabulate(GDIM)(i => Array.tabulate(GDIM)(j => (i + j) % 7 + 0.5)))

  // ─── 1. Vector Add ───────────────────────────────────────────────────────
  banner("VectorD  +  (N=10M)")
  DeviceConfig.useGPU = true;  timed("GPU") { v1 + v2 }
  DeviceConfig.useGPU = false; timed("CPU") { v1 + v2 }

  // ─── 2. Vector Subtract ──────────────────────────────────────────────────
  banner("VectorD  -  (N=10M)")
  DeviceConfig.useGPU = true;  timed("GPU") { v1 - v2 }
  DeviceConfig.useGPU = false; timed("CPU") { v1 - v2 }

  // ─── 3. Vector Multiply ──────────────────────────────────────────────────
  banner("VectorD  *  (N=10M)")
  DeviceConfig.useGPU = true;  timed("GPU") { v1 * v2 }
  DeviceConfig.useGPU = false; timed("CPU") { v1 * v2 }

  // ─── 4. Vector Divide ────────────────────────────────────────────────────
  banner("VectorD  /  (N=10M)")
  DeviceConfig.useGPU = true;  timed("GPU") { v1 / v2 }
  DeviceConfig.useGPU = false; timed("CPU") { v1 / v2 }

  // ─── 5. Vector Sum ───────────────────────────────────────────────────────
  banner("VectorD  sum  (N=10M)")
  DeviceConfig.useGPU = true
  val gpuSum = timed("GPU") { v1.sum }
  DeviceConfig.useGPU = false
  val cpuSum = timed("CPU") { v1.sum }
  println(s"  GPU=$gpuSum  CPU=$cpuSum")

  // ─── 6. Vector Min ───────────────────────────────────────────────────────
  banner("VectorD  min  (N=10M)")
  DeviceConfig.useGPU = true
  val gpuMin = timed("GPU") { v1.min }
  DeviceConfig.useGPU = false
  val cpuMin = timed("CPU") { v1.min }
  println(s"  GPU=$gpuMin  CPU=$cpuMin")

  // ─── 7. Vector Max ───────────────────────────────────────────────────────
  banner("VectorD  max  (N=10M)")
  DeviceConfig.useGPU = true
  val gpuMax = timed("GPU") { v1.max }
  DeviceConfig.useGPU = false
  val cpuMax = timed("CPU") { v1.max }
  println(s"  GPU=$gpuMax  CPU=$cpuMax")

  // ─── 8. Vector Dot Product ───────────────────────────────────────────────
  banner("VectorD  dot  (N=10M)")
  DeviceConfig.useGPU = true
  val gpuDot = timed("GPU") { v1 dot v2 }
  DeviceConfig.useGPU = false
  val cpuDot = timed("CPU") { v1 dot v2 }
  println(s"  GPU=$gpuDot  CPU=$cpuDot")

  // ─── 9. Vector NormSq ────────────────────────────────────────────────────
  banner("VectorD  normSq  (N=10M)")
  DeviceConfig.useGPU = true
  val gpuNsq = timed("GPU") { v1.normSq }
  DeviceConfig.useGPU = false
  val cpuNsq = timed("CPU") { v1.normSq }
  println(s"  GPU=$gpuNsq  CPU=$cpuNsq")

  // ─── 10. Vector Norm ─────────────────────────────────────────────────────
  // GPU path: normSq on device then sqrt on host; same FLOP count as normSq.
  banner("VectorD  norm  (N=10M)")
  DeviceConfig.useGPU = true
  val gpuNorm = timed("GPU") { v1.norm }
  DeviceConfig.useGPU = false
  val cpuNorm = timed("CPU") { v1.norm }
  println(s"  GPU=$gpuNorm  CPU=$cpuNorm")

  // ─── 11. Vector Norm1 ────────────────────────────────────────────────────
  banner("VectorD  norm1  (N=10M)")
  DeviceConfig.useGPU = true
  val gpuN1 = timed("GPU") { v1.norm1 }
  DeviceConfig.useGPU = false
  val cpuN1 = timed("CPU") { v1.norm1 }
  println(s"  GPU=$gpuN1  CPU=$cpuN1")

  // ─── 12. GEMM  (MatrixD * MatrixD) ───────────────────────────────────────
  // GPU kernel only supports square matrices; non-square falls through to CPU.
  banner(s"MatrixD  *  GEMM  (${GDIM}×${GDIM})")
  DeviceConfig.useGPU = true
  val gpuGemm = timed("GPU") { g1 * g2 }
  DeviceConfig.useGPU = false
  val cpuGemm = timed("CPU") { g1 * g2 }
  println(s"  GPU(0,0)=${gpuGemm(0,0)}  CPU(0,0)=${cpuGemm(0,0)}")

  // ─── 13. GEMM via mul (transpose-based) ──────────────────────────────────
  banner(s"MatrixD  mul  GEMM  (${GDIM}×${GDIM})")
  DeviceConfig.useGPU = true;  timed("GPU") { g1.mul( g2 ) }
  DeviceConfig.useGPU = false; timed("CPU") { g1.mul( g2 )}

  // ─── 14. GEMV  (MatrixD * VectorD) ───────────────────────────────────────
  banner(s"MatrixD  *  GEMV  (${DIM}×${DIM})")
  DeviceConfig.useGPU = true
  val gpuGemv = timed("GPU") { m1 * v_m }
  DeviceConfig.useGPU = false
  val cpuGemv = timed("CPU") { m1 * v_m }
  println(s"  GPU(0)=${gpuGemv(0)}  CPU(0)=${cpuGemv(0)}")

  // ─── 15. Column Sum ──────────────────────────────────────────────────────
  banner(s"MatrixD  sumV  col-sum  (${DIM}×${DIM})")
  DeviceConfig.useGPU = true
  val gpuCsum = timed("GPU") { m1.sumV }
  DeviceConfig.useGPU = false
  val cpuCsum = timed("CPU") { m1.sumV }
  println(s"  GPU(0)=${gpuCsum(0)}  CPU(0)=${cpuCsum(0)}")

  // ─── 16. Column Min ──────────────────────────────────────────────────────
  banner(s"MatrixD  min  col-min  (${DIM}×${DIM})")
  DeviceConfig.useGPU = true
  val gpuCmin = timed("GPU") { m1.min }
  DeviceConfig.useGPU = false
  val cpuCmin = timed("CPU") { m1.min }
  println(s"  GPU(0)=${gpuCmin(0)}  CPU(0)=${cpuCmin(0)}")

  // ─── 17. Column Max ──────────────────────────────────────────────────────
  banner(s"MatrixD  max  col-max  (${DIM}×${DIM})")
  DeviceConfig.useGPU = true
  val gpuCmax = timed("GPU") { m1.max }
  DeviceConfig.useGPU = false
  val cpuCmax = timed("CPU") { m1.max }
  println(s"  GPU(0)=${gpuCmax(0)}  CPU(0)=${cpuCmax(0)}")

  DeviceConfig.useGPU = true   // restore default
  println("\nAll benchmarks complete.")
