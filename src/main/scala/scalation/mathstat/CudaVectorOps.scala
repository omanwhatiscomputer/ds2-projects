package scalation.mathstat

import java.lang.foreign.*
import java.lang.invoke.MethodHandle
import java.lang.foreign.ValueLayout.*

object CudaVectorOps:

  // Load the shared libraries (adjust paths if needed)
  private val libCheck   = SymbolLookup.libraryLookup("src/main/scala/scalation/mathstat/libC/libcudacheck.so", Arena.global())
  private val libKernels = SymbolLookup.libraryLookup("src/main/scala/scalation/mathstat/libC/libcudakernels.so", Arena.global())

  // Get the linker (used to create downcall handles)
  private val linker = Linker.nativeLinker()

  // ---- CUDA availability check (once) ----
  private val cudaIsAvailableAddr: MemorySegment =
    libCheck.find("cuda_is_available")
      .orElseThrow(() => new RuntimeException("Cannot find cuda_is_available"))

  private val cudaIsAvailable: MethodHandle =
    linker.downcallHandle(
      cudaIsAvailableAddr,
      FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN) // returns bool
    )

  private val _available: Boolean =
    try
      // Use invoke() instead of invokeExact() to allow boxing/unboxing
      val avail = cudaIsAvailable.invoke().asInstanceOf[Boolean]
//      println(s"[CudaVectorOps] CUDA available check returned: $avail")
      avail
    catch case e: Throwable =>
      println(s"[CudaVectorOps] CUDA availability check threw: ${e.getMessage}")
      e.printStackTrace()
      false

  def isAvailable: Boolean = _available

  // op codes matching the CUDA kernel switch statement
  private val OP_ADD = 0
  private val OP_SUB = 1
  private val OP_MUL = 2
  private val OP_DIV = 3

  // ---- Single address for the combined kernel ----
  private val opAddr: MemorySegment =
    libKernels.find("gpuVectorOp")
      .orElseThrow(() => new RuntimeException("Cannot find gpuVectorOp"))

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

  private val opName = Map(OP_ADD -> "add", OP_SUB -> "sub", OP_MUL -> "mul", OP_DIV -> "div")

  // ---- Shared dispatch logic ----
  private def dispatch(a: Array[Double], b: Array[Double], op: Int): Option[Array[Double]] =
    if !isAvailable then return None
    val n = a.length
    if n != b.length then return None
    val result = new Array[Double](n)
    try
      invokeKernel(opAddr, a, b, result, n, op)
      // println(s"[CudaVectorOps] GPU executed: ${opName(op)} on $n elements")
      Some(result)
    catch case e: Throwable =>
      e.printStackTrace()
      None

  // ---- Public API ----
  def add(a: Array[Double], b: Array[Double]): Option[Array[Double]] = dispatch(a, b, OP_ADD)
  def sub(a: Array[Double], b: Array[Double]): Option[Array[Double]] = dispatch(a, b, OP_SUB)
  def mul(a: Array[Double], b: Array[Double]): Option[Array[Double]] = dispatch(a, b, OP_MUL)
  def div(a: Array[Double], b: Array[Double]): Option[Array[Double]] = dispatch(a, b, OP_DIV)
end CudaVectorOps