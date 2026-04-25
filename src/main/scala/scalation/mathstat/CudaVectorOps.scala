package scalation.mathstat

import java.lang.foreign.*
//import java.lang.invoke.MethodHandle
import java.lang.foreign.ValueLayout.*
import scala.util.Try

object CudaVectorOps:

  // Load the shared libraries (adjust paths if needed)
  private val libCheck   = Try(SymbolLookup.libraryLookup("src/main/scala/scalation/mathstat/libC/libcudacheck.so", Arena.global())).toOption
  private val libKernels = Try(SymbolLookup.libraryLookup("src/main/scala/scalation/mathstat/libC/libcudakernels.so", Arena.global())).toOption

  // Get the linker (used to create downcall handles)
  private val linker = Linker.nativeLinker()

  // ---- CUDA availability check (once) ----
  private val _available: Boolean =
    libCheck match
      case None => false
      case Some(lib) =>
        try
          val cudaIsAvailableAddr = lib.find("cuda_is_available")
            .orElseThrow(() => new RuntimeException("Cannot find cuda_is_available"))
          val cudaIsAvailable = linker.downcallHandle(
            cudaIsAvailableAddr,
            FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN)
          )
          // Use invoke() instead of invokeExact() to allow boxing/unboxing
          cudaIsAvailable.invoke().asInstanceOf[Boolean]
        catch case e: Throwable =>
          println(s"[CudaVectorOps] CUDA availability check threw: ${e.getMessage}")
          false

  def isAvailable: Boolean = DeviceConfig.useGPU && _available

  // op codes matching the CUDA kernel switch statement
  private val OP_ADD = 0
  private val OP_SUB = 1
  private val OP_MUL = 2
  private val OP_DIV = 3

  // ---- Addresses for the combined kernels ----
  // lazy: only evaluated on first dispatch call, which is guarded by isAvailable
  private lazy val opAddr: MemorySegment =
    libKernels.get.find("gpuVectorOp")
      .orElseThrow(() => new RuntimeException("Cannot find gpuVectorOp"))

  private lazy val scalarOpAddr: MemorySegment =
    libKernels.get.find("gpuVectorScalarOp")
      .orElseThrow(() => new RuntimeException("Cannot find gpuVectorScalarOp"))

  // ---- Generic helper to invoke a kernel that takes two input arrays and one output array ----
  private def invokeKernel(
                            kernelAddr: MemorySegment,
                            a: Array[Double],
                            b: Array[Double],
                            result: Array[Double],
                            n: Int,
                            op: Int
                          ): Unit =
    // void gpuVectorOp(double*, double*, double*, int n, int op)
    val kernelHandle = linker.downcallHandle(
      kernelAddr,
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT)
    )

    val byteSize = n * 8L
    val arena = Arena.ofConfined()
    try
      val segA   = arena.allocate(byteSize)
      val segB   = arena.allocate(byteSize)
      val segRes = arena.allocate(byteSize)

      MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_DOUBLE, 0, n)
      MemorySegment.copy(b, 0, segB, ValueLayout.JAVA_DOUBLE, 0, n)

      kernelHandle.invoke(segA, segB, segRes, n, op)

      MemorySegment.copy(segRes, ValueLayout.JAVA_DOUBLE, 0, result, 0, n)
    finally
      arena.close()
    end try
  end invokeKernel
  // DEBUG line below
//   private val opName = Map(OP_ADD -> "add", OP_SUB -> "sub", OP_MUL -> "mul", OP_DIV -> "div")

  // ---- Shared dispatch logic ----
  private def dispatch(a: Array[Double], b: Array[Double], op: Int): Option[Array[Double]] =
    if !isAvailable then return None
    val n = a.length
    if n != b.length then return None
    val result = new Array[Double](n)
    try
      invokeKernel(opAddr, a, b, result, n, op)
//       println(s"[CudaVectorOps] GPU executed: ${opName(op)} on $n elements")
      Some(result)
    catch case e: Throwable =>
      e.printStackTrace()
      None

  // ---- Scalar kernel invoke helper ----
  private def invokeScalarKernel(a: Array[Double], scalar: Double, result: Array[Double], n: Int, op: Int): Unit =
    // void gpuVectorScalarOp(double*, double scalar, double*, int n, int op)
    val kernelHandle = linker.downcallHandle(
      scalarOpAddr,
      FunctionDescriptor.ofVoid(ADDRESS, JAVA_DOUBLE, ADDRESS, JAVA_INT, JAVA_INT)
    )
    val byteSize = n * 8L
    val arena = Arena.ofConfined()
    try
      val segA   = arena.allocate(byteSize)
      val segRes = arena.allocate(byteSize)
      MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_DOUBLE, 0, n)
      kernelHandle.invoke(segA, scalar, segRes, n, op)
      MemorySegment.copy(segRes, ValueLayout.JAVA_DOUBLE, 0, result, 0, n)
    finally
      arena.close()
    end try
  end invokeScalarKernel

  // ---- Shared scalar dispatch logic ----
  private def dispatchScalar(a: Array[Double], scalar: Double, op: Int): Option[Array[Double]] =
    if !isAvailable then return None
    val n = a.length
    val result = new Array[Double](n)
    try
      invokeScalarKernel(a, scalar, result, n, op)
//       println(s"[CudaVectorOps] GPU executed: ${opName(op)} scalar on $n elements")
      Some(result)
    catch case e: Throwable =>
      e.printStackTrace()
      None

  // ---- Public API (vector op vector) ----
  def add(a: Array[Double], b: Array[Double]): Option[Array[Double]] = dispatch(a, b, OP_ADD)
  def sub(a: Array[Double], b: Array[Double]): Option[Array[Double]] = dispatch(a, b, OP_SUB)
  def mul(a: Array[Double], b: Array[Double]): Option[Array[Double]] = dispatch(a, b, OP_MUL)
  def div(a: Array[Double], b: Array[Double]): Option[Array[Double]] = dispatch(a, b, OP_DIV)

  // ---- Public API (vector op scalar) ----
  def addScalar(a: Array[Double], s: Double): Option[Array[Double]] = dispatchScalar(a, s, OP_ADD)
  def subScalar(a: Array[Double], s: Double): Option[Array[Double]] = dispatchScalar(a, s, OP_SUB)
  def mulScalar(a: Array[Double], s: Double): Option[Array[Double]] = dispatchScalar(a, s, OP_MUL)
  def divScalar(a: Array[Double], s: Double): Option[Array[Double]] = dispatchScalar(a, s, OP_DIV)

  // ---- Vector reduction kernel addresses ----
  // lazy: guarded by isAvailable before any dispatch call
  private lazy val vectorSumAddr:   MemorySegment = libKernels.get.find("gpuVectorSum").orElseThrow(() => new RuntimeException("Cannot find gpuVectorSum"))
  private lazy val vectorMinAddr:   MemorySegment = libKernels.get.find("gpuVectorMin").orElseThrow(() => new RuntimeException("Cannot find gpuVectorMin"))
  private lazy val vectorMaxAddr:   MemorySegment = libKernels.get.find("gpuVectorMax").orElseThrow(() => new RuntimeException("Cannot find gpuVectorMax"))
  private lazy val vectorDotAddr:   MemorySegment = libKernels.get.find("gpuVectorDot").orElseThrow(() => new RuntimeException("Cannot find gpuVectorDot"))
  private lazy val vectorNormSqAddr: MemorySegment = libKernels.get.find("gpuVectorNormSq").orElseThrow(() => new RuntimeException("Cannot find gpuVectorNormSq"))
  private lazy val vectorNormAddr:  MemorySegment = libKernels.get.find("gpuVectorNorm").orElseThrow(() => new RuntimeException("Cannot find gpuVectorNorm"))
  private lazy val vectorNorm1Addr: MemorySegment = libKernels.get.find("gpuVectorNorm1").orElseThrow(() => new RuntimeException("Cannot find gpuVectorNorm1"))

  // ---- Invoke helper: void(ptr_a, ptr_result, int n) → scalar ----
  private def invokeScalarReduction(kernelAddr: MemorySegment, a: Array[Double]): Double =
    val kernelHandle = linker.downcallHandle(
      kernelAddr,
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT)
    )
    val n     = a.length
    val arena = Arena.ofConfined()
    try
      val segA   = arena.allocate(n.toLong * 8)
      val segRes = arena.allocate(8L)
      MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_DOUBLE, 0, n)
      kernelHandle.invoke(segA, segRes, n)
      segRes.get(ValueLayout.JAVA_DOUBLE, 0)
    finally
      arena.close()

  // ---- Invoke helper: void(ptr_a, ptr_b, ptr_result, int n) → scalar (dot) ----
  private def invokeDot(a: Array[Double], b: Array[Double]): Double =
    val kernelHandle = linker.downcallHandle(
      vectorDotAddr,
      FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, JAVA_INT)
    )
    val n     = a.length
    val arena = Arena.ofConfined()
    try
      val segA   = arena.allocate(n.toLong * 8)
      val segB   = arena.allocate(n.toLong * 8)
      val segRes = arena.allocate(8L)
      MemorySegment.copy(a, 0, segA, ValueLayout.JAVA_DOUBLE, 0, n)
      MemorySegment.copy(b, 0, segB, ValueLayout.JAVA_DOUBLE, 0, n)
      kernelHandle.invoke(segA, segB, segRes, n)
      segRes.get(ValueLayout.JAVA_DOUBLE, 0)
    finally
      arena.close()

  // ---- Shared dispatch for scalar reductions ----
  private def dispatchReduction(kernelAddr: MemorySegment, a: Array[Double]): Option[Double] =
    if !isAvailable then return None
    try Some(invokeScalarReduction(kernelAddr, a))
    catch case e: Throwable => e.printStackTrace(); None

  // ---- Public API (vector reductions → scalar) ----
  def sum(a: Array[Double]):   Option[Double] = dispatchReduction(vectorSumAddr, a)
  def min(a: Array[Double]):   Option[Double] = dispatchReduction(vectorMinAddr, a)
  def max(a: Array[Double]):   Option[Double] = dispatchReduction(vectorMaxAddr, a)
  def normSq(a: Array[Double]): Option[Double] = dispatchReduction(vectorNormSqAddr, a)
  def norm(a: Array[Double]):  Option[Double] = dispatchReduction(vectorNormAddr, a)
  def norm1(a: Array[Double]): Option[Double] = dispatchReduction(vectorNorm1Addr, a)
  def dot(a: Array[Double], b: Array[Double]): Option[Double] =
    if !isAvailable then return None
    if a.length != b.length then return None
    try Some(invokeDot(a, b))
    catch case e: Throwable => e.printStackTrace(); None
end CudaVectorOps