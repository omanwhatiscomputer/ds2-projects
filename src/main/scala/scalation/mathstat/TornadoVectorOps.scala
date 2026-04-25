package scalation
package mathstat

import scala.util.Try

import uk.ac.manchester.tornado.api.enums.DataTransferMode
import uk.ac.manchester.tornado.api.annotations.{Parallel, Reduce}
import uk.ac.manchester.tornado.api.common.TornadoFunctions
import uk.ac.manchester.tornado.api.{TaskGraph, TornadoExecutionPlan}

// %%% TornadoVM kernel methods %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
object TVKernels:

  def add(a: Array[Double], b: Array[Double], c: Array[Double]): Unit =
    @Parallel var i = 0
    while i < a.length do { c(i) = a(i) + b(i); i += 1 }

  def sub(a: Array[Double], b: Array[Double], c: Array[Double]): Unit =
    @Parallel var i = 0
    while i < a.length do { c(i) = a(i) - b(i); i += 1 }

  def mul(a: Array[Double], b: Array[Double], c: Array[Double]): Unit =
    @Parallel var i = 0
    while i < a.length do { c(i) = a(i) * b(i); i += 1 }

  def div(a: Array[Double], b: Array[Double], c: Array[Double]): Unit =
    @Parallel var i = 0
    while i < a.length do { c(i) = a(i) / b(i); i += 1 }

  def sum(a: Array[Double], @Reduce result: Array[Double]): Unit =
    result(0) = 0.0
    @Parallel var i = 0
    while i < a.length do { result(0) += a(i); i += 1 }

  def min(a: Array[Double], @Reduce result: Array[Double]): Unit =
    result(0) = Double.MaxValue
    @Parallel var i = 0
    while i < a.length do { result(0) = Math.min(result(0), a(i)); i += 1 }

  def max(a: Array[Double], @Reduce result: Array[Double]): Unit =
    result(0) = -Double.MaxValue
    @Parallel var i = 0
    while i < a.length do { result(0) = Math.max(result(0), a(i)); i += 1 }

  def dot(a: Array[Double], b: Array[Double], @Reduce result: Array[Double]): Unit =
    result(0) = 0.0
    @Parallel var i = 0
    while i < a.length do { result(0) += a(i) * b(i); i += 1 }

  def normSq(a: Array[Double], @Reduce result: Array[Double]): Unit =
    result(0) = 0.0
    @Parallel var i = 0
    while i < a.length do { result(0) += a(i) * a(i); i += 1 }

  def norm1(a: Array[Double], @Reduce result: Array[Double]): Unit =
    result(0) = 0.0
    @Parallel var i = 0
    while i < a.length do { result(0) += Math.abs(a(i)); i += 1 }

// %%% Public dispatch API %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// Returns Some(result) on GPU success, None to signal CPU fallback.
object TornadoVectorOps:

  def isAvailable: Boolean = DeviceConfig.useGPU

  // %% Element-wise (returns flat result array) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  def add(a: Array[Double], b: Array[Double]): Option[Array[Double]] = elemOp2("tvo_add", TVKernels.add, a, b)
  def sub(a: Array[Double], b: Array[Double]): Option[Array[Double]] = elemOp2("tvo_sub", TVKernels.sub, a, b)
  def mul(a: Array[Double], b: Array[Double]): Option[Array[Double]] = elemOp2("tvo_mul", TVKernels.mul, a, b)
  def div(a: Array[Double], b: Array[Double]): Option[Array[Double]] = elemOp2("tvo_div", TVKernels.div, a, b)

  // %% Reductions (returns scalar) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  def sum(a: Array[Double]): Option[Double]  = scalarOp1("tvo_sum",  TVKernels.sum,   a, 0.0)
  def min(a: Array[Double]): Option[Double]  = scalarOp1("tvo_min",  TVKernels.min,   a, Double.MaxValue)
  def max(a: Array[Double]): Option[Double]  = scalarOp1("tvo_max",  TVKernels.max,   a, -Double.MaxValue)
  def normSq(a: Array[Double]): Option[Double] = scalarOp1("tvo_nsq", TVKernels.normSq, a, 0.0)
  def norm1(a: Array[Double]): Option[Double]  = scalarOp1("tvo_n1",  TVKernels.norm1, a, 0.0)

  def norm(a: Array[Double]): Option[Double] = normSq(a).map(Math.sqrt)

  def dot(a: Array[Double], b: Array[Double]): Option[Double] =
    if !isAvailable then return None
    val result = Array(0.0)
    Try {
      setDevice("tvo_dot")
      val kDot: TornadoFunctions.Task3[Array[Double], Array[Double], Array[Double]] = TVKernels.dot
      val plan = new TornadoExecutionPlan(
        new TaskGraph("tvo_dot")
          .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b, result)
          .task("t0", kDot, a, b, result)
          .transferToHost(DataTransferMode.EVERY_EXECUTION, result)
          .snapshot())
      plan.execute()
      result(0)
    }.toOption

  // %% Private helpers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  private def setDevice(graphName: String): Unit =
    DeviceConfig.applyDeviceProperty(graphName, "t0")

  // Use TornadoVM's own SAM types so Scala converts method references correctly.
  private type Task3D = TornadoFunctions.Task3[Array[Double], Array[Double], Array[Double]]
  private type Task2D = TornadoFunctions.Task2[Array[Double], Array[Double]]

  private def elemOp2(name: String, k: Task3D, a: Array[Double], b: Array[Double]): Option[Array[Double]] =
    if !isAvailable then return None
    val c = new Array[Double](a.length)
    Try {
      setDevice(name)
      val plan = new TornadoExecutionPlan(
        new TaskGraph(name)
          .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, b)
          .task("t0", k, a, b, c)
          .transferToHost(DataTransferMode.EVERY_EXECUTION, c)
          .snapshot())
      plan.execute()
      c
    }.toOption

  private def scalarOp1(name: String, k: Task2D, a: Array[Double], init: Double): Option[Double] =
    if !isAvailable then return None
    val result = Array(init)
    Try {
      setDevice(name)
      val plan = new TornadoExecutionPlan(
        new TaskGraph(name)
          .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, result)
          .task("t0", k, a, result)
          .transferToHost(DataTransferMode.EVERY_EXECUTION, result)
          .snapshot())
      plan.execute()
      result(0)
    }.toOption
