package project3

import scalation.mathstat.VectorD

import uk.ac.manchester.tornado.api.enums.DataTransferMode
import uk.ac.manchester.tornado.api.annotations.Parallel
import uk.ac.manchester.tornado.api.{TaskGraph, TornadoExecutionPlan}

object Kernels:
  // 1. The Kernel MUST use primitive arrays and primitive loops
  def vectorAdd(a: Array[Double], b: Array[Double], c: Array[Double]): Unit = {
    // Put the @Parallel annotation directly on the loop variable definition!
    @Parallel var i = 0
    while (i < a.length) do {
      c(i) = a(i) + b(i)
      i += 1
    }
  }

@main def TestRunner(): Unit =
  val size = 1_000_000

  // 2. Initialize raw primitive arrays for the hardware
  val a = Array.tabulate(size)(i => i.toDouble)
  val b = Array.fill(size)(2.0)
  val c = new Array[Double](size)

  // 3. Define the hardware execution plan
  val taskGraph = new TaskGraph("s0")
    .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
    .task("t0", Kernels.vectorAdd, a, b, c)
    .transferToHost(DataTransferMode.EVERY_EXECUTION, c)

  val plan = new TornadoExecutionPlan(taskGraph.snapshot())

  println("Offloading computation to TornadoVM...")
  plan.execute() // This triggers the JIT compilation and GPU execution

  // 4. Wrap the GPU's result back into a sKalation VectorD for downstream use
  val v3 = VectorD(c)

  // Verify it worked
  println(s"First 3 elements of v3: ${v3(0)}, ${v3(1)}, ${v3(2)}")
  if v3(0) == 2.0 then println("SUCCESS: Hardware execution complete!")